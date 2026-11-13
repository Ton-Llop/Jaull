"""Firewall rules are generated for the administrator, limited to the local subnet."""

from __future__ import annotations

from jaull.application.serving.security import firewall_rule, subnet_of


def test_windows_rule_opens_only_the_port_to_the_local_subnet() -> None:
    rule = firewall_rule("windows", 8090)

    assert rule.startswith("New-NetFirewallRule ")
    assert "-LocalPort 8090" in rule
    assert "-RemoteAddress LocalSubnet" in rule
    assert "-Direction Inbound" in rule
    assert "Public" not in rule


def test_linux_rule_uses_the_detected_subnet() -> None:
    assert firewall_rule("linux", 8090, subnet="192.168.1.0/24") == (
        "sudo ufw allow from 192.168.1.0/24 to any port 8090 proto tcp"
    )


def test_linux_rule_without_subnet_asks_for_it() -> None:
    assert "<your-local-subnet" in firewall_rule("linux", 8090)


def test_subnet_from_address_and_netmask() -> None:
    assert subnet_of("192.168.1.107", "255.255.255.0") == "192.168.1.0/24"
    assert subnet_of("10.2.3.4", "255.255.0.0") == "10.2.0.0/16"
