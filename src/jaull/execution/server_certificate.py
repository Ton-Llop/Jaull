"""A self-signed certificate, so a server shared on the local network can speak HTTPS.

llama-server only reads a certificate and a key; it never makes them. A self-signed one
encrypts the traffic but is not trusted by anything on its own: browsers warn once, and
a client has to be told to trust the file.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from jaull.execution.server_process import ServerProcessError
from jaull.runtime import policies

CERTIFICATE_FILENAME = "server-cert.pem"
KEY_FILENAME = "server-key.pem"
# Always covered, so the same certificate also works from the PC that runs the server.
_THIS_PC_ADDRESSES = ("127.0.0.1",)
_THIS_PC_NAMES = ("localhost",)


@dataclass(frozen=True)
class ServerCertificate:
    cert_path: Path
    key_path: Path
    #: Addresses first, then names, as written in the certificate.
    names: tuple[str, ...]
    expires: datetime
    #: False when an existing certificate was kept.
    created: bool
    #: Addresses and names asked for now that an existing certificate does not cover.
    missing: tuple[str, ...]


def has_server_certificate(directory: Path) -> bool:
    """Whether a certificate and its key are already in ``directory``."""
    return (directory / CERTIFICATE_FILENAME).is_file() and (directory / KEY_FILENAME).is_file()


def ensure_server_certificate(
    directory: Path,
    *,
    addresses: Sequence[str],
    hostnames: Sequence[str],
    renew: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ServerCertificate:
    """The certificate in ``directory``, made when it is missing, expired or ``renew`` asks.

    An existing one is kept even if this PC's address changed: replacing it would silently
    break every client that already trusts it, so what it lacks is reported in ``missing``.
    """
    wanted = _unique([*addresses, *_THIS_PC_ADDRESSES, *hostnames, *_THIS_PC_NAMES])
    cert_path = directory / CERTIFICATE_FILENAME
    key_path = directory / KEY_FILENAME
    if not renew and has_server_certificate(directory):
        existing = _read(cert_path)
        if existing.not_valid_after_utc > now():
            covered = _names(existing)
            return ServerCertificate(
                cert_path=cert_path, key_path=key_path, names=covered,
                expires=existing.not_valid_after_utc, created=False,
                missing=tuple(name for name in wanted if name.lower() not in
                              {item.lower() for item in covered}),
            )
    created = _create(wanted, now())
    directory.mkdir(parents=True, exist_ok=True)
    certificate, key = created
    _write_private(key_path, key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return ServerCertificate(
        cert_path=cert_path, key_path=key_path, names=_names(certificate),
        expires=certificate.not_valid_after_utc, created=True, missing=(),
    )


def _create(
    wanted: Sequence[str], started: datetime,
) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, policies.LLAMA_SERVER_CERTIFICATE_NAME),
    ])
    backdate = timedelta(hours=policies.LLAMA_SERVER_CERTIFICATE_BACKDATE_HOURS)
    entries: list[x509.GeneralName] = []
    for item in wanted:
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(item)))
        except ValueError:
            continue
    entries += [x509.DNSName(item) for item in wanted if not _is_address(item)]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(started - backdate)
        .not_valid_after(started + timedelta(days=policies.LLAMA_SERVER_CERTIFICATE_DAYS))
        .add_extension(x509.SubjectAlternativeName(entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    return certificate, key


def _read(path: Path) -> x509.Certificate:
    try:
        return x509.load_pem_x509_certificate(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ServerProcessError(
            f"The certificate {path} cannot be read. Run `jaull serve cert --renew` to "
            "replace it."
        ) from exc


def _names(certificate: x509.Certificate) -> tuple[str, ...]:
    try:
        san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return ()
    return (
        *(str(item) for item in san.get_values_for_type(x509.IPAddress)),
        *san.get_values_for_type(x509.DNSName),
    )


def _is_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _unique(items: Sequence[str]) -> list[str]:
    seen: dict[str, str] = {}
    for item in items:
        if item and item.lower() not in seen:
            seen[item.lower()] = item
    return list(seen.values())


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


__all__ = ["ServerCertificate", "ensure_server_certificate", "has_server_certificate"]
