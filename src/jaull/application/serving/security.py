"""Firewall rules for sharing a server on the local network.

Jaull only writes the rule for the administrator to read and run: changing the
firewall is a system security setting, so it is never applied silently.
"""

from __future__ import annotations

import ipaddress

# Shown before and after starting a server shared on the network without HTTPS.
LAN_WITHOUT_TLS_WARNING = (
    "Shared without HTTPS: prompts, answers and the API key cross the local network "
    "unencrypted, so anyone on it who can capture traffic can read them. Give a "
    "certificate and a key to serve over HTTPS."
)
# Slots are easy to read as accounts; they are not.
SLOTS_ARE_NOT_USERS = (
    "Slots are parallel lanes, not separate users: everyone shares one API key, any "
    "request can take any free slot, and nothing keeps one person's conversations "
    "apart from another's."
)
# What a certificate Jaull makes itself does and does not give.
SELF_SIGNED_CERTIFICATE_NOTE = (
    "Self-signed: the traffic is encrypted, but browsers and programs do not trust it by "
    "themselves. A browser shows a warning to accept once; to avoid it, make each client "
    "trust the certificate file."
)
# Measured on the B60 (docs/TFG/concurrencia-b60.md): llama-server queues nothing when the
# shared pool fills, it ends every running request with "Context size has been exceeded".
SHARED_POOL_OVERFLOW_WARNING = (
    "If the conversations together outgrow the pool, llama-server stops all of them at "
    "once, not only the longest."
)
# What the prompt cache switch changes, in both directions.
PROMPT_CACHE_NOTE = {
    True: (
        "Faster replies: the start of each conversation is kept and reused on the next "
        "turn, and llama.cpp keeps idle conversations in up to 8 GiB of system RAM. That "
        "content stays in the server's memory after a chat ends, and a reply that starts "
        "sooner can show that someone else sent the same beginning."
    ),
    False: (
        "Nothing is kept between requests, so finished conversations leave nothing in "
        "memory. Every turn reprocesses the whole conversation: long chats answer much "
        "more slowly, from seconds to tens of seconds per turn on long contexts."
    ),
}


def firewall_rule(os_name: str, port: int, *, subnet: str | None = None) -> str:
    """The command that opens ``port`` to the local network only.

    Windows resolves ``LocalSubnet`` itself. Linux needs the subnet in CIDR form;
    without one the rule says so instead of guessing.
    """
    if os_name == "windows":
        return (
            f'New-NetFirewallRule -DisplayName "Jaull llama-server {port}" '
            f"-Direction Inbound -Protocol TCP -LocalPort {port} "
            "-RemoteAddress LocalSubnet -Profile Domain,Private -Action Allow"
        )
    source = subnet or "<your-local-subnet, e.g. 192.168.1.0/24>"
    return f"sudo ufw allow from {source} to any port {port} proto tcp"


def subnet_of(address: str, netmask: str) -> str:
    """``192.168.1.107`` + ``255.255.255.0`` -> ``192.168.1.0/24``."""
    return str(ipaddress.IPv4Interface(f"{address}/{netmask}").network)


__all__ = [
    "LAN_WITHOUT_TLS_WARNING",
    "PROMPT_CACHE_NOTE",
    "SELF_SIGNED_CERTIFICATE_NOTE",
    "SHARED_POOL_OVERFLOW_WARNING",
    "SLOTS_ARE_NOT_USERS",
    "firewall_rule",
    "subnet_of",
]
