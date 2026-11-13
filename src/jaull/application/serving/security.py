"""Firewall rules for sharing a server on the local network.

Jaull only writes the rule for the administrator to read and run: changing the
firewall is a system security setting, so it is never applied silently.
"""

from __future__ import annotations

import ipaddress


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


__all__ = ["firewall_rule", "subnet_of"]
