"""Export public Windows trust certificates for the campaign HTTP clients."""

import ssl
from pathlib import Path

import certifi

target = Path('.venv/rtx4060-trust.pem')
if target.exists():
    raise SystemExit('Refusing to overwrite an existing certificate bundle')
certificates = [Path(certifi.where()).read_text(encoding='ascii')]
count = 0
for store in ('ROOT', 'CA'):
    for certificate, encoding, trust in ssl.enum_certificates(store):
        if encoding == 'x509_asn' and (trust is True or ssl.Purpose.SERVER_AUTH.oid in trust):
            certificates.append(ssl.DER_cert_to_PEM_cert(certificate))
            count += 1
target.write_text('\n'.join(certificates), encoding='ascii')
print(f'Public system certificates: {count}')
print(f'Trust bundle: {target.resolve()}')
