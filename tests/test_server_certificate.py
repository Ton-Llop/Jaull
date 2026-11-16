"""The self-signed certificate `jaull serve cert` writes for HTTPS on the local network."""

from __future__ import annotations

import ipaddress
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtendedKeyUsageOID

from jaull.execution.server_certificate import ensure_server_certificate
from jaull.runtime import policies

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def _make(tmp_path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    defaults: dict[str, object] = {
        "addresses": ["192.168.1.107"], "hostnames": ["monk-e"], "now": lambda: NOW,
    }
    return ensure_server_certificate(tmp_path / "tls", **{**defaults, **kwargs})  # type: ignore[arg-type]


def _load(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def test_the_certificate_covers_the_lan_address_the_hostname_and_this_pc(
    tmp_path: Path,
) -> None:
    result = _make(tmp_path)

    san = _load(result.cert_path).extensions.get_extension_for_class(
        x509.SubjectAlternativeName,
    ).value
    assert set(san.get_values_for_type(x509.IPAddress)) == {
        ipaddress.ip_address("192.168.1.107"), ipaddress.ip_address("127.0.0.1"),
    }
    assert set(san.get_values_for_type(x509.DNSName)) == {"monk-e", "localhost"}
    assert result.created and result.missing == ()
    assert set(result.names) == {"192.168.1.107", "127.0.0.1", "monk-e", "localhost"}


def test_it_is_a_server_certificate_valid_for_the_policy_period(tmp_path: Path) -> None:
    certificate = _load(_make(tmp_path).cert_path)

    assert certificate.not_valid_before_utc <= NOW
    assert certificate.not_valid_after_utc == NOW + timedelta(
        days=policies.LLAMA_SERVER_CERTIFICATE_DAYS,
    )
    assert not certificate.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    usage = certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert list(usage) == [ExtendedKeyUsageOID.SERVER_AUTH]


def test_the_key_matches_the_certificate_and_only_its_owner_reads_it(tmp_path: Path) -> None:
    result = _make(tmp_path)

    key = serialization.load_pem_private_key(result.key_path.read_bytes(), password=None)
    certificate_key = _load(result.cert_path).public_key()
    assert key.public_key().public_numbers() == certificate_key.public_numbers()  # type: ignore[union-attr,attr-defined]
    if sys.platform != "win32":
        assert stat.S_IMODE(result.key_path.stat().st_mode) == 0o600


def test_asking_again_keeps_the_certificate_clients_already_trust(tmp_path: Path) -> None:
    first = _make(tmp_path)
    second = _make(tmp_path)

    assert not second.created
    assert second.cert_path.read_bytes() == first.cert_path.read_bytes()
    assert second.expires == first.expires


def test_a_new_address_is_reported_not_silently_replaced(tmp_path: Path) -> None:
    _make(tmp_path)

    moved = _make(tmp_path, addresses=["192.168.1.50"])

    assert not moved.created
    assert moved.missing == ("192.168.1.50",)


def test_renewing_replaces_it_with_one_that_covers_the_new_address(tmp_path: Path) -> None:
    before = _make(tmp_path).cert_path.read_bytes()

    renewed = _make(tmp_path, addresses=["192.168.1.50"], renew=True)

    assert renewed.created and renewed.missing == ()
    assert "192.168.1.50" in renewed.names
    assert renewed.cert_path.read_bytes() != before


def test_an_expired_certificate_is_replaced(tmp_path: Path) -> None:
    before = _make(tmp_path).cert_path.read_bytes()
    later = NOW + timedelta(days=policies.LLAMA_SERVER_CERTIFICATE_DAYS + 1)

    fresh = _make(tmp_path, now=lambda: later)

    assert fresh.created
    assert fresh.expires == later + timedelta(days=policies.LLAMA_SERVER_CERTIFICATE_DAYS)
    assert fresh.cert_path.read_bytes() != before


def test_an_unreadable_certificate_is_an_error_not_a_silent_new_one(tmp_path: Path) -> None:
    from jaull.execution.server_process import ServerProcessError

    result = _make(tmp_path)
    result.cert_path.write_text("not a certificate")

    with pytest.raises(ServerProcessError, match="--renew"):
        _make(tmp_path)


def test_a_certificate_is_found_only_when_both_files_exist(tmp_path: Path) -> None:
    from jaull.execution.server_certificate import has_server_certificate

    assert not has_server_certificate(tmp_path / "tls")
    result = _make(tmp_path)
    assert has_server_certificate(tmp_path / "tls")
    result.key_path.unlink()
    assert not has_server_certificate(tmp_path / "tls")


def test_the_certificate_lasts_a_year() -> None:
    # Not rotated often on purpose: the key sits next to an API key that never expires, and a
    # certificate that keeps changing teaches people to accept the browser warning.
    assert policies.LLAMA_SERVER_CERTIFICATE_DAYS == 365
