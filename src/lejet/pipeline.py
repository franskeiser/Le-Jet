"""
pipeline.py -- orchestration of one inspection (plan section 13.1) and of a whole batch.

One URL, in order (plan 13.1):
    Stage 0  normalize + safety gate + the 14 URL-text features        (no network)
    Stage 1  DNS of the ORIGINAL host (liveness)
    Stage 3  HTTP fetch, following the chain by hand; HTML analysed in the parse pool
    Stage 1/1b DNS record + RDAP of the FINAL registrable domain (plan 13.2b)
    Stage 5  the 12 HTML features come back from the parse pool
    Stage 6  persist (storage.py)

A failure in one stage never aborts the later independent stages, and every URL yields exactly
one row with every feature key present (-1 = unknown, never omitted).

Features describe the FINAL URL (the page the victim lands on), except the 14 URL-text
features, which describe the ORIGINAL link exactly as the SMS contained it (parity with the
training data). The `orig_*` columns keep the original-domain values so either choice can be
audited later.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

from lejet import domains, safety, schema, urlnorm
from lejet.features import external_features
from lejet.parse_pool import ParsePool
from lejet.settings import Settings
from lejet.stages.dns_stage import DnsStage, DnspythonResolver, Resolver
from lejet.stages.http_stage import DomainLocks, Hop, HttpStage
from lejet.stages import ct_stage
from lejet.stages.rdap_stage import Bootstrap, HttpxRdapFetcher, RdapClient
from lejet.storage import Storage

_link_features = None


def _load_link_features():
    """The 14 URL-text features come from src/features/link_features.py -- they are NOT
    re-implemented (plan 13.2, 14.0.1). Loaded once; a missing module is a loud error."""
    global _link_features
    if _link_features is None:
        try:
            from src.features import link_features as lf          # type: ignore
        except ImportError as exc:                                 # pragma: no cover
            raise RuntimeError(
                "cannot import src.features.link_features: the repository root must be on "
                "sys.path (and pandas, numpy, tldextract, nostril installed)"
            ) from exc
        _link_features = lf
    return _link_features


@dataclass
class Inspection:
    row: Dict[str, Any]
    hops: List[Hop] = field(default_factory=list)
    errors: List[Tuple[str, str, str]] = field(default_factory=list)
    normalized_url: str = ""


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None, *,
                 resolver: Optional[Resolver] = None,
                 rdap_fetch=None, bootstrap: Optional[Bootstrap] = None,
                 popular_domains: Optional[Set[str]] = None, ct_fetch=None):
        self.s = settings or Settings()
        self._resolver = resolver
        self._rdap_fetch = rdap_fetch
        self._ct_fetch = ct_fetch            # tests inject a fake; settings.ct_enabled turns the stage on
        self.ct: Optional[ct_stage.CtClient] = None
        self._own_ct_fetcher: Optional[ct_stage.HttpxCtFetcher] = None
        self._bootstrap = bootstrap
        self._popular = popular_domains
        self.pool: Optional[ParsePool] = None
        self.dns: Optional[DnsStage] = None
        self.rdap: Optional[RdapClient] = None
        self.http: Optional[HttpStage] = None
        self._own_fetcher: Optional[HttpxRdapFetcher] = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> "Pipeline":
        _load_link_features()                                   # fail loudly, early
        s = self.s
        self.pool = await ParsePool(s.parse_workers, s.parse_timeout).start()
        self.dns = DnsStage(self._resolver or DnspythonResolver(s.dns_server))
        bootstrap = self._bootstrap
        if bootstrap is None:
            bootstrap = Bootstrap.from_file(s.rdap_bootstrap_path) if s.rdap_bootstrap_path else Bootstrap([])
        fetch = self._rdap_fetch
        if fetch is None:
            self._own_fetcher = HttpxRdapFetcher()
            fetch = self._own_fetcher
        self.rdap = RdapClient(bootstrap, fetch, min_interval=s.rdap_min_interval)
        if s.rdap_cache_path:
            self.rdap.load_cache(s.rdap_cache_path)
        if s.ct_enabled or self._ct_fetch is not None:
            ct_fetch = self._ct_fetch
            if ct_fetch is None:
                self._own_ct_fetcher = ct_stage.HttpxCtFetcher()
                ct_fetch = self._own_ct_fetcher
            self.ct = ct_stage.CtClient(ct_fetch, min_interval=s.ct_min_interval,
                                        retry_backoff=s.ct_backoff)
            if s.ct_cache_path:
                self.ct.load_cache(s.ct_cache_path)
            if s.ct_probe:
                await self.ct.probe()           # a dead crt.sh pauses certificate lookups from the start
        if self._popular is None:
            self._popular = _load_domain_list(s.popular_domains_path)
        self.http = HttpStage(s, self.pool, dns=self.dns, locks=DomainLocks())
        return self

    async def close(self) -> None:
        if self.rdap is not None and self.s.rdap_cache_path:
            try:
                self.rdap.save_cache(self.s.rdap_cache_path)
            except OSError:
                pass
        if self.ct is not None and self.s.ct_cache_path:
            try:
                self.ct.save_cache(self.s.ct_cache_path)
            except OSError:
                pass
        if self._own_ct_fetcher is not None:
            await self._own_ct_fetcher.aclose()
        if self._own_fetcher is not None:
            await self._own_fetcher.aclose()
        if self.pool is not None:
            await self.pool.close()

    async def __aenter__(self) -> "Pipeline":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------------ one URL
    async def inspect(self, url_id: str, raw_url: str, batch_id: str) -> Inspection:
        started = time.monotonic()
        insp = Inspection(row=schema.new_row(url_id, raw_url, batch_id))
        try:
            await self._inspect(insp, raw_url)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                                # noqa: BLE001
            insp.row["fail_reason"] = "internal_error"
            insp.errors.append(("pipeline", type(exc).__name__, str(exc)))
        row = insp.row
        row["fetch_ok"] = 1 if row["fail_reason"] == "none" else 0
        row["page_fetched"] = row["html_available"]
        row["duration_ms"] = _ms(started)
        return insp

    def _url_text_features(self, raw_url: str, insp: Inspection) -> None:
        lf = _load_link_features()
        feats = lf.no_link_features()
        if len(raw_url) <= safety.MAX_URL_LEN:
            try:
                feats = lf.link_text_features(lf.normalize_link(raw_url))
            except Exception as exc:                            # noqa: BLE001
                insp.errors.append(("stage0", type(exc).__name__, "link_text_features"))
        insp.row.update({name: feats.get(name, -1) for name in schema.URL_TEXT_FEATURES})

    async def _domain_features(self, host: str, reg_domain: str, path: str,
                               insp: Inspection) -> Dict[str, Any]:
        """dns_record + the RDAP features + hosting flags for one host (plan 13.4b, 13.2c)."""
        assert self.dns is not None and self.rdap is not None
        out: Dict[str, Any] = {}
        t = time.monotonic()
        out["dns_record"] = await self.dns.dns_record(host, reg_domain)
        insp.row["stage_ms_dns"] += _ms(t)

        hosting = domains.classify_hosting(host, path)
        out["free_hosting_subdomain"] = hosting.free_hosting_subdomain
        out["hosting_platform"] = hosting.hosting_platform
        out["platform_path_hosting"] = hosting.platform_path_hosting
        out["rdap_skipped"] = 1 if hosting.skip_rdap else 0
        out.update(ct_first_cert_days=-1, ct_ok=0, registration_unavailable_tld=0)

        if hosting.skip_rdap or not reg_domain or domains.host_is_ip(host):
            out.update(external_features.not_looked_up())
            return out
        t = time.monotonic()
        outcome = await self.rdap.lookup(reg_domain, domains.registry_tld(host))
        insp.row["stage_ms_rdap"] += _ms(t)
        out.update(external_features.features_from_record(outcome.record))
        if outcome.note and outcome.note not in ("fetch_failed",):
            insp.errors.append(("rdap", "note", outcome.note))
        if outcome.note == "no_rdap_server":
            out["registration_unavailable_tld"] = 1
        if not outcome.record.ok and self.ct is not None:      # RDAP gave nothing: certificate age
            t = time.monotonic()
            ct = await self.ct.lookup(reg_domain)
            insp.row["stage_ms_rdap"] += _ms(t)
            out["ct_ok"] = 1 if ct.record.ok else 0
            out["ct_first_cert_days"] = ct_stage.age_days(ct.record)
            if ct.note and ct.note not in ("fetch_failed", "ct_paused"):
                insp.errors.append(("ct", "note", ct.note))
        return out

    async def _inspect(self, insp: Inspection, raw_url: str) -> None:
        assert self.dns is not None and self.http is not None
        row = insp.row
        s = self.s

        # ---- Stage 0 ------------------------------------------------------------------
        norm = urlnorm.normalize(raw_url, allow_loopback=s.allow_loopback)
        self._url_text_features(raw_url, insp)
        if not norm.ok:
            row["fail_reason"] = norm.fail_reason or "rejected_url"
            return
        insp.normalized_url = norm.normalized_url or ""
        orig_host = norm.host_ascii or ""
        orig_reg = norm.registrable_domain or ""
        row["reg_domain"] = orig_reg or orig_host
        row["punycode"] = int(norm.punycode)
        orig_path = urlsplit(insp.normalized_url).path

        # ---- Stage 1: DNS of the original host (liveness) --------------------------------
        t = time.monotonic()
        if s.allow_loopback and orig_host in urlnorm.LOOPBACK_TEST_HOSTS:
            dns_ok, dns_fail = True, "none"                     # test mode: fixtures on loopback
        else:
            dres = await self.dns.resolve_host(orig_host)
            dns_ok, dns_fail = dres.resolves, dres.fail_reason()
            row["stage_ms_dns"] += _ms(t)
        row["dns_ok"] = int(dns_ok)
        row["stage_reached"] = 1

        # ---- Stage 3: HTTP (only if the host resolves) -----------------------------------
        final_url = insp.normalized_url
        http = None
        if dns_ok:
            http = await self.http.fetch(norm)
            insp.hops = http.hops
            row["stage_ms_http"] = max(0, http.elapsed_ms - http.html_ms)
            row["stage_ms_html"] = http.html_ms
            row["stage_reached"] = 3
            row["fail_reason"] = http.fail_reason
            row["hop_count"] = http.redirects
            row["chain_complete"] = int(http.chain_complete)
            row["scheme_used"] = http.scheme_used
            row["final_status"] = http.final_status
            row["body_truncated"] = int(http.body_truncated)
            final_url = http.final_url or final_url
            for note in http.errors:
                insp.errors.append(("http", "note", note))
        else:
            row["fail_reason"] = dns_fail if dns_fail != "none" else "dns_no_answer"

        # ---- the FINAL url and its domain (plan 13.2b) --------------------------------------
        final_host, final_reg, final_path = orig_host, orig_reg, orig_path
        if final_url != insp.normalized_url:
            fnorm = urlnorm.normalize(final_url, allow_loopback=s.allow_loopback)
            if fnorm.ok:
                final_host = fnorm.host_ascii or orig_host
                final_reg = fnorm.registrable_domain or ""
                final_path = urlsplit(final_url).path
        row["final_url"] = final_url
        row["final_reg_domain"] = final_reg or final_host
        row["final_differs"] = int((final_reg or final_host) != (orig_reg or orig_host))

        # ---- DNS record + RDAP: original first, final only if it changed ---------------------
        loopback_test = s.allow_loopback and orig_host in urlnorm.LOOPBACK_TEST_HOSTS
        if loopback_test:
            orig_ext = dict(external_features.not_looked_up(), dns_record=-1,
                            free_hosting_subdomain=0, hosting_platform="",
                            platform_path_hosting=0, rdap_skipped=0, ct_first_cert_days=-1,
                            ct_ok=0, registration_unavailable_tld=0)
        else:
            orig_ext = await self._domain_features(orig_host, orig_reg, orig_path, insp)
        final_ext = orig_ext
        if (final_host, final_reg) != (orig_host, orig_reg) and not (
                s.allow_loopback and final_host in urlnorm.LOOPBACK_TEST_HOSTS):
            final_ext = await self._domain_features(final_host, final_reg, final_path, insp)

        for name in schema.EXTERNAL_FEATURES:
            row[name] = final_ext[name]
        for name in ("days_to_expiry", "rdap_ok", "free_hosting_subdomain", "hosting_platform",
                     "platform_path_hosting", "rdap_skipped", "ct_first_cert_days", "ct_ok",
                     "registration_unavailable_tld"):
            row[name] = final_ext[name] if name in final_ext else row[name]
        row["orig_whois_registered_domain"] = orig_ext["whois_registered_domain"]
        row["orig_domain_registration_length"] = orig_ext["domain_registration_length"]
        row["orig_domain_age"] = orig_ext["domain_age"]
        row["orig_dns_record"] = orig_ext["dns_record"]

        popular = bool(final_reg) and final_reg in (self._popular or ())
        row["final_is_popular"] = int(popular and not final_ext["platform_path_hosting"])

        # ---- Stage 5: the 12 HTML features (computed in the parse pool during Stage 3) -------
        if http is not None and http.analysis is not None and row["fail_reason"] == "none":
            feats = http.analysis.get("features") or {}
            if feats:
                for name in schema.WEBPAGE_FEATURES:
                    row[name] = feats.get(name, -1)
                row["html_available"] = 1
                row["stage_reached"] = 5
            row["html_bytes"] = int(http.analysis.get("html_bytes", -1))
            row["script_count"] = int(http.analysis.get("script_count", -1))
            row["near_empty"] = int(http.analysis.get("near_empty", -1))
        elif http is not None and http.analysis is not None:     # challenge / parked page
            row["html_bytes"] = int(http.analysis.get("html_bytes", -1))
            row["script_count"] = int(http.analysis.get("script_count", -1))
            row["near_empty"] = int(http.analysis.get("near_empty", -1))
        if row["fail_reason"] == "none" and not row["html_available"]:
            row["fail_reason"] = "blocked_or_empty"             # parsed, but nothing usable

    # ------------------------------------------------------------------ a whole batch
    async def run(self, items: Sequence[Tuple[str, str]], batch_id: str, store: Storage,
                  progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, int]:
        """Inspect every (url_id, raw_url), resuming over what is already in `store`."""
        s = self.s
        done = store.done_ids()
        todo = sorted((it for it in items if it[0] not in done), key=lambda it: it[0])
        rng = random.Random(s.seed)
        rng.shuffle(todo)                                        # seeded shuffle (plan 13.2)
        sem = asyncio.Semaphore(max(1, s.concurrency))
        interval = 1.0 / s.rate if s.rate > 0 else 0.0
        next_start = 0.0
        tasks: Set["asyncio.Task[None]"] = set()
        finished = 0
        total = len(todo)

        async def one(url_id: str, raw: str) -> None:
            nonlocal finished
            try:
                insp = await self.inspect(url_id, raw, batch_id)
                store.save(insp.row, insp.hops, insp.errors, insp.normalized_url)
            finally:
                sem.release()
                finished += 1
                if progress is not None:
                    progress(finished, total)

        try:
            for url_id, raw in todo:
                await sem.acquire()
                now = time.monotonic()
                wait = max(0.0, next_start - now) + (rng.uniform(0, s.jitter) if s.jitter > 0 else 0.0)
                if wait > 0:
                    await asyncio.sleep(wait)
                next_start = time.monotonic() + interval
                task = asyncio.create_task(one(url_id, raw))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            if tasks:
                await asyncio.gather(*list(tasks))
        except BaseException:
            for t in list(tasks):
                t.cancel()
            await asyncio.gather(*list(tasks), return_exceptions=True)
            raise
        return store.status_counts(batch_id)


def _load_domain_list(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    try:
        with open(path, encoding="utf-8") as fh:
            return {ln.strip().lower() for ln in fh if ln.strip() and not ln.startswith("#")}
    except OSError:
        return set()


def config_hash(settings: Settings) -> str:
    blob = json.dumps(settings.__dict__, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
