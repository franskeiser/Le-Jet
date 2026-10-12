"""
dns_stage.py -- Stage 1: DNS (plan section 13.3).

Resolves through the gateway's Unbound (10.77.0.1) with dnspython: 3 s timeout, 1 retry.
The resolver is behind a tiny interface (`Resolver.query`) so the stage can be tested with a
fake that needs no network.

`dns_record` direction (same as the public dataset, plan 14.0.4):
    0 = the host / its registrable domain has DNS records (A, AAAA or NS) -- the NORMAL case
    1 = NXDOMAIN / no records                                           -- the abnormal case
   -1 = the resolver failed or timed out
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from lejet import domains, safety


@dataclass(frozen=True)
class RawAnswer:
    """One DNS query result, independent of dnspython."""
    status: str                    # ok | nxdomain | nodata | timeout | servfail | error
    values: Tuple[str, ...] = ()
    ttl: int = -1


class Resolver(Protocol):
    async def query(self, name: str, rdtype: str) -> RawAnswer: ...


@dataclass
class DnsResult:
    status: str                    # ok | nxdomain | timeout | servfail | no_answer | private_blocked | error
    addresses: List[str] = field(default_factory=list)
    min_ttl: int = -1
    ms: int = 0
    private_answer_blocked: bool = False

    @property
    def resolves(self) -> bool:
        return self.status == "ok" and bool(self.addresses)

    def fail_reason(self) -> str:
        """The row's fail_reason when the host did not resolve, else 'none'."""
        return {
            "ok": "none",
            "nxdomain": "dns_nxdomain",
            "timeout": "dns_timeout",
            "servfail": "dns_servfail",
            "no_answer": "dns_no_answer",
            "private_blocked": "dns_no_answer",     # logged as no_answer (plan 13.3)
        }.get(self.status, "dns_servfail")


class DnspythonResolver:
    """The real resolver: dnspython pointed at one nameserver (the gateway)."""

    def __init__(self, nameserver: str = "10.77.0.1", timeout: float = 3.0, lifetime: float = 7.0):
        import dns.asyncresolver  # imported here so the module loads without dnspython

        r = dns.asyncresolver.Resolver(configure=False)
        r.nameservers = [nameserver]
        r.timeout = timeout          # per attempt
        r.lifetime = lifetime        # total: one retry fits in
        r.use_edns(0, 0, 1232)
        self._resolver = r

    async def query(self, name: str, rdtype: str) -> RawAnswer:
        import dns.exception
        import dns.resolver

        try:
            answer = await self._resolver.resolve(name, rdtype, raise_on_no_answer=False)
        except dns.resolver.NXDOMAIN:
            return RawAnswer("nxdomain")
        except dns.resolver.NoAnswer:
            return RawAnswer("nodata")
        except dns.exception.Timeout:
            return RawAnswer("timeout")
        except dns.resolver.NoNameservers:
            return RawAnswer("servfail")
        except dns.exception.DNSException:
            return RawAnswer("error")
        except Exception:                                    # noqa: BLE001
            return RawAnswer("error")
        rrset = answer.rrset
        if rrset is None or len(rrset) == 0:
            return RawAnswer("nodata")
        values = tuple(
            r.to_text() if rdtype in ("NS", "MX", "TXT") else r.address
            for r in rrset if rdtype in ("NS", "MX", "TXT") or hasattr(r, "address")
        )
        return RawAnswer("ok", values, int(rrset.ttl))


class DnsStage:
    def __init__(self, resolver: Resolver):
        self._resolver = resolver
        self._host_cache: Dict[str, DnsResult] = {}
        self._ns_cache: Dict[str, str] = {}

    async def resolve_host(self, host: str) -> DnsResult:
        """A and AAAA for one host, with the rebinding backstop (every answer is checked
        again with safety.is_ip_allowed). Results are cached per host."""
        key = domains.clean_host(host)
        if key in self._host_cache:
            return self._host_cache[key]
        started = time.monotonic()
        result = await self._resolve_uncached(key)
        result.ms = int((time.monotonic() - started) * 1000)
        self._host_cache[key] = result
        return result

    async def _resolve_uncached(self, host: str) -> DnsResult:
        if domains.host_is_ip(host):
            ok = safety.is_ip_allowed(host)
            return DnsResult("ok" if ok else "private_blocked", [host] if ok else [],
                             private_answer_blocked=not ok)
        a, aaaa = await asyncio.gather(self._resolver.query(host, "A"),
                                       self._resolver.query(host, "AAAA"))
        raw: List[str] = [v for ans in (a, aaaa) if ans.status == "ok" for v in ans.values]
        allowed = [ip for ip in raw if safety.is_ip_allowed(ip)]
        ttls = [ans.ttl for ans in (a, aaaa) if ans.status == "ok" and ans.ttl >= 0]
        min_ttl = min(ttls) if ttls else -1
        if allowed:
            return DnsResult("ok", allowed, min_ttl, private_answer_blocked=len(allowed) < len(raw))
        if raw:                                               # answers existed, all private
            return DnsResult("private_blocked", [], min_ttl, private_answer_blocked=True)
        statuses = {a.status, aaaa.status}
        if "nxdomain" in statuses:
            return DnsResult("nxdomain")
        if "timeout" in statuses:
            return DnsResult("timeout")
        if "servfail" in statuses:
            return DnsResult("servfail")
        if "error" in statuses:
            return DnsResult("error")
        return DnsResult("no_answer")

    async def dns_record(self, host: str, reg_domain: str) -> int:
        """The `dns_record` feature for a host and its registrable domain (0 / 1 / -1)."""
        res = await self.resolve_host(host)
        if res.resolves or res.status == "private_blocked":
            return 0                                          # it has records (even if unusable)
        if res.status in ("timeout", "servfail", "error"):
            return -1
        # NXDOMAIN or no address: the registrable domain may still have NS records
        domain = domains.clean_host(reg_domain)
        if not domain:
            return 1
        if domain not in self._ns_cache:
            ns = await self._resolver.query(domain, "NS")
            self._ns_cache[domain] = (
                "has" if ns.status == "ok" and ns.values
                else "none" if ns.status in ("nxdomain", "nodata")
                else "fail"
            )
        state = self._ns_cache[domain]
        return 0 if state == "has" else 1 if state == "none" else -1
