"""azure_run.py -- ONE live link, inspected on the Azure VM (the cloud lab used by the Le-Jet tool / frontend).

    python -I -m lejet.azure_run <batch_id>

What it does (the same steps and the same code as the resident worker in the VirtualBox lab, run once, no spool):
    fetch (Pipeline.inspect) -> database -> screenshots of the WHOLE page + up to 4 pages of the same site, with the
    "open every site" retry ladder -> export (<batch>.jsonl.gz, .manifest.json, .shots.tar, sha256 sidecars).

Where the files are (same layout as the VirtualBox lab; the program runs as the unprivileged user `fetcher`):
    /var/lib/lejet/in/urls.txt      one line  url_id<TAB>url  (installed by the root helper lejet-azure-live; the
                                    url_id is re-checked here)
    /var/lib/lejet/out/export/<batch>.*   the result files the helper hands to the SSH user; the PC pulls them
                                    (the VM never connects back)

Safety, from outside in (this VM has no VPN gateway like the VirtualBox lab, so the barriers are different):
    * `infra/azure/egress.nft` (loaded by setup_shot_vm.sh) rejects every connection of user `fetcher` to private
      ranges, the Azure metadata service and the Azure host address; `infra/azure/gate.sh` writes the token this
      program checks (`/run/lejet-preflight-ok`) only while that rule is loaded and really blocks the metadata
      address. No token = no run;
    * the Chromium sandbox stays on, every page goes through its own guard proxy (public addresses, ports 80/443 only),
      downloads / popups / dialogs / permissions are blocked, one empty browser context per page (shot_stage.py);
    * the page's JavaScript only ever runs inside that browser; nothing here executes anything the page sends;
    * the host validates everything it pulls (validate_output.py, validate_shots.py) before it is used.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import List, Optional

from lejet.cli import DEFAULT_TOKEN
from lejet.live_worker import ID_RE, LiveWorker, settings_for_vm

BASE = "/var/lib/lejet"
INPUT = os.path.join(BASE, "in", "urls.txt")
OUT = os.path.join(BASE, "out")
SPOOL = os.path.join(BASE, "work", "spool")            # LiveWorker wants a spool folder; this one-shot run never uses it
TOKEN_MAX_AGE = 900                                    # the gate timer renews the token every few minutes
PUBLIC_DNS = os.environ.get("LEJET_DNS", "1.1.1.1")    # the VirtualBox lab's 10.77.0.1 does not exist here
EXTRA_WAIT_MS = int(os.environ.get("LEJET_SHOT_EXTRA_WAIT_MS", "6000"))


class AzureWorker(LiveWorker):
    """The resident worker's request handler, run once; it also prints the "== n ..." progress lines the tool shows."""

    def _stage(self, batch: str, stage: str) -> None:
        super()._stage(batch, stage)
        if stage == "shot":
            print("== 4b whole-page screenshots (and a few pages of the same site)", flush=True)


def tunnel_resolver(nameserver: str = PUBLIC_DNS):
    """Look names up by asking `nameserver` directly (UDP). Run as user `fetcher`, that traffic goes through the VPN tunnel,
    so the names the browser visits are not looked up by the cloud provider's resolver. IPv4 only: the tunnel carries no IPv6."""
    async def resolve(host: str, port: int):
        import dns.asyncresolver
        r = dns.asyncresolver.Resolver(configure=False)
        r.nameservers, r.lifetime = [nameserver], 6.0
        answer = await r.resolve(host, "A")
        return [rr.address for rr in answer]
    return resolve


def settings_for_azure():
    s = settings_for_vm(OUT)
    s.dns_server = PUBLIC_DNS
    bootstrap = "/opt/lejet/rdap-dns.json"
    s.rdap_bootstrap_path = bootstrap if os.path.exists(bootstrap) else None
    return s


async def amain(batch: str) -> int:
    from lejet.pipeline import Pipeline
    from lejet.stages.shot_stage import ShotRunner

    settings = settings_for_azure()
    os.makedirs(OUT, exist_ok=True)
    extras = {"git_commit": os.environ.get("LEJET_GIT_COMMIT", ""), "vbox_version": "azure",
              "vpn_provider": "", "vpn_exit_country": ""}
    vpn_on = os.environ.get("LEJET_VPN_ACTIVE") == "1"          # set by lejet-azure-live when the Mullvad tunnel is installed
    if vpn_on:
        extras.update({"vpn_provider": os.environ.get("LEJET_VPN_PROVIDER", ""),
                       "vpn_exit_country": os.environ.get("LEJET_VPN_EXIT_COUNTRY", "")})
    async with Pipeline(settings) as pipeline:
        # whole-site mode: tall picture of the page, 4 same-site pages, and the retry ladder for stubborn sites
        async with ShotRunner(os.path.join(OUT, "export", "shots-live"), mobile=True, realistic=True,
                              extra_wait_ms=EXTRA_WAIT_MS, full_page=True, max_pages=4, ladder=True,
                              resolver=tunnel_resolver() if vpn_on else None) as shots:
            worker = AzureWorker(pipeline, shots, settings=settings, spool=SPOOL, input_path=INPUT, out_dir=OUT,
                                token=DEFAULT_TOKEN, token_max_age=TOKEN_MAX_AGE, extras=extras,
                                snapshot_name="azure-lejetshot", shot_deadline=420.0)
            payload = await worker.handle(batch)
    print("== result", payload, flush=True)
    return 0 if payload.get("ok") else 5


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or not ID_RE.match(args[0]):
        print("usage: python -m lejet.azure_run <batch_id>   (letters, digits, - and _ only)", file=sys.stderr)
        return 2
    if os.geteuid() == 0:                              # type: ignore[attr-defined]
        print("REFUSING TO START: do not run page-opening code as root", file=sys.stderr)
        return 3
    return asyncio.run(amain(args[0]))


if __name__ == "__main__":
    sys.exit(main())
