"""
Tests for Stage 1, DNS (WEB_INSPECTION_PLAN.md sections 13.3, 14.0.4), with a fake resolver:
no network.

dns_record direction (same as the public dataset): 0 = the host/domain HAS DNS records (normal),
1 = NXDOMAIN/no records (abnormal), -1 = the resolver failed.

Run with:  python -m pytest tests/test_dns_stage.py
"""

from __future__ import annotations

import asyncio

import pytest

from lejet.stages.dns_stage import DnsStage, RawAnswer


class FakeResolver:
    def __init__(self, table=None, default="nxdomain"):
        self.table = table or {}
        self.default = default
        self.calls = []

    async def query(self, name, rdtype):
        self.calls.append((name, rdtype))
        ans = self.table.get((name, rdtype))
        if ans is None:
            return RawAnswer(self.default)
        return ans


def ok(*values, ttl=300):
    return RawAnswer("ok", tuple(values), ttl)


def run(coro):
    return asyncio.run(coro)


def test_a_record_resolves_and_dns_record_is_zero():
    stage = DnsStage(FakeResolver({("example.com", "A"): ok("93.184.216.34", ttl=120)}, default="nodata"))
    res = run(stage.resolve_host("example.com"))
    assert res.resolves and res.status == "ok"
    assert res.addresses == ["93.184.216.34"] and res.min_ttl == 120
    assert res.fail_reason() == "none"
    assert run(stage.dns_record("example.com", "example.com")) == 0


def test_aaaa_only_host_resolves():
    stage = DnsStage(FakeResolver({("v6.example.com", "AAAA"): ok("2606:4700::1111")}, default="nodata"))
    res = run(stage.resolve_host("v6.example.com"))
    assert res.resolves and res.addresses == ["2606:4700::1111"]


def test_nxdomain_gives_dns_record_one_and_fail_reason():
    stage = DnsStage(FakeResolver(default="nxdomain"))
    res = run(stage.resolve_host("nope.example.org"))
    assert res.status == "nxdomain" and not res.resolves
    assert res.fail_reason() == "dns_nxdomain"
    assert run(stage.dns_record("nope.example.org", "example.org")) == 1      # abnormal case


def test_host_missing_but_domain_has_ns_records_is_normal():
    stage = DnsStage(FakeResolver({("example.org", "NS"): ok("ns1.example.org.")}, default="nxdomain"))
    assert run(stage.dns_record("www.example.org", "example.org")) == 0


def test_no_a_no_ns_is_abnormal():
    stage = DnsStage(FakeResolver({("example.org", "NS"): RawAnswer("nodata")}, default="nodata"))
    assert run(stage.dns_record("www.example.org", "example.org")) == 1


@pytest.mark.parametrize("status, reason", [("timeout", "dns_timeout"), ("servfail", "dns_servfail")])
def test_resolver_trouble_is_minus_one(status, reason):
    stage = DnsStage(FakeResolver(default=status))
    res = run(stage.resolve_host("slow.example.com"))
    assert res.status == status and res.fail_reason() == reason
    assert run(stage.dns_record("slow.example.com", "example.com")) == -1


def test_a_timeout_on_one_family_does_not_hide_a_good_answer():
    stage = DnsStage(FakeResolver({("h.example.com", "A"): ok("93.184.216.34"),
                                   ("h.example.com", "AAAA"): RawAnswer("timeout")}))
    assert run(stage.resolve_host("h.example.com")).resolves


def test_empty_answer_is_dns_no_answer():
    stage = DnsStage(FakeResolver(default="nodata"))
    res = run(stage.resolve_host("empty.example.com"))
    assert res.status == "no_answer" and res.fail_reason() == "dns_no_answer"


@pytest.mark.parametrize("private", ["10.0.0.5", "192.168.1.1", "127.0.0.1", "169.254.169.254",
                                     "172.16.5.5", "100.64.1.1", "0.0.0.0", "::1", "fe80::1", "fc00::1"])
def test_private_answers_are_blocked_as_a_rebinding_backstop(private):
    rdtype = "AAAA" if ":" in private else "A"
    stage = DnsStage(FakeResolver({("rebind.example.com", rdtype): ok(private)}, default="nodata"))
    res = run(stage.resolve_host("rebind.example.com"))
    assert not res.resolves and res.status == "private_blocked" and res.private_answer_blocked
    assert res.fail_reason() == "dns_no_answer"
    assert run(stage.dns_record("rebind.example.com", "example.com")) == 0   # records exist


def test_mixed_public_and_private_answers_keep_only_the_public_ones():
    stage = DnsStage(FakeResolver({("mix.example.com", "A"): ok("93.184.216.34", "10.0.0.5")}, default="nodata"))
    res = run(stage.resolve_host("mix.example.com"))
    assert res.resolves and res.addresses == ["93.184.216.34"] and res.private_answer_blocked


def test_ip_literal_hosts_need_no_lookup():
    fake = FakeResolver()
    stage = DnsStage(fake)
    assert run(stage.resolve_host("93.184.216.34")).resolves
    assert not run(stage.resolve_host("10.1.2.3")).resolves
    assert not run(stage.resolve_host("[::1]")).resolves
    assert fake.calls == []


def test_results_are_cached_per_host():
    fake = FakeResolver({("c.example.com", "A"): ok("93.184.216.34")}, default="nodata")
    stage = DnsStage(fake)
    run(stage.resolve_host("c.example.com"))
    n = len(fake.calls)
    run(stage.resolve_host("C.Example.Com."))
    assert len(fake.calls) == n


def test_ns_lookup_is_cached_per_registrable_domain():
    fake = FakeResolver({("example.org", "NS"): ok("ns1.example.org.")}, default="nxdomain")
    stage = DnsStage(fake)

    async def go():
        await stage.dns_record("a.example.org", "example.org")
        await stage.dns_record("b.example.org", "example.org")
    run(go())
    assert sum(1 for c in fake.calls if c == ("example.org", "NS")) == 1


def test_ns_failure_is_minus_one():
    fake = FakeResolver({("example.org", "NS"): RawAnswer("timeout")}, default="nxdomain")
    assert run(DnsStage(fake).dns_record("www.example.org", "example.org")) == -1


def test_no_registrable_domain_and_no_address_is_abnormal():
    assert run(DnsStage(FakeResolver(default="nxdomain")).dns_record("nonexistent", "")) == 1
