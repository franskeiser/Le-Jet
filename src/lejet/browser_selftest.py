"""
browser_selftest.py -- does headless Chromium work here, with its sandbox ON?

    python -I -m lejet.browser_selftest

Run as the unprivileged user `fetcher` by infra/analysis/install_browser.sh (and by you, any time).
It launches Chromium exactly the way the screenshot stage does (`chromium_sandbox=True`, our argument
list), renders a tiny inline page, checks that the PNG is structurally valid, and then reads
/proc to prove that NO Chromium process was started with a sandbox-disabling switch.

Exit code 0 and the line `BROWSER SELFTEST OK` mean the browser works and its sandbox is enabled.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from lejet import pngcheck
from lejet.stages import shot_stage

_BAD = ("--no-sandbox", "--disable-setuid-sandbox", "--single-process", "--disable-gpu-sandbox",
        "--disable-seccomp-filter-sandbox")


def chromium_cmdlines() -> list:
    """The command lines of all running processes that look like Chromium (Linux /proc)."""
    out = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                parts = fh.read().decode("utf-8", "replace").split("\x00")
        except OSError:
            continue
        joined = " ".join(parts)
        if "chrom" in (parts[0] if parts else "") or "headless_shell" in joined or "chrome-headless-shell" in joined:
            out.append(parts)
    return out


async def _run() -> dict:
    from playwright.async_api import async_playwright

    info = {"launched": False, "png_ok": False, "sandbox_on": False, "processes": 0, "reason": ""}
    assert shot_stage.chromium_args_ok(shot_stage.CHROMIUM_ARGS)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, chromium_sandbox=True,
                                           args=list(shot_stage.CHROMIUM_ARGS), timeout=60_000)
        info["launched"] = True
        info["version"] = browser.version
        context = await browser.new_context(viewport=shot_stage.VIEWPORT, java_script_enabled=True)
        page = await context.new_page()
        await page.set_content("<html><body style='background:#fff'><h1>Le-Jet browser self-test</h1>"
                               "<p>if you can read this in the screenshot, rendering works</p></body></html>")
        data = await page.screenshot(type="png", full_page=False)
        ok, why, w, h = pngcheck.check_png(data)
        info["png_ok"], info["png_size"] = ok, [w, h]
        if not ok:
            info["reason"] = f"png: {why}"
        cmdlines = chromium_cmdlines()
        info["processes"] = len(cmdlines)
        bad = [c for c in cmdlines if any(a.split("=", 1)[0] in _BAD for a in c)]
        info["sandbox_on"] = bool(cmdlines) and not bad
        if bad:
            info["reason"] = "a Chromium process runs with a sandbox-disabling switch"
        elif not cmdlines:
            info["reason"] = "no Chromium process found in /proc (cannot prove the sandbox is on)"
        await context.close()
        await browser.close()
    return info


def main() -> int:
    try:
        info = asyncio.run(_run())
    except Exception as exc:                                          # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:300]}"}))
        print("BROWSER SELFTEST FAILED")
        return 1
    print(json.dumps(info, sort_keys=True))
    if info["launched"] and info["png_ok"] and info["sandbox_on"]:
        print("BROWSER SELFTEST OK")
        return 0
    print("BROWSER SELFTEST FAILED: " + (info.get("reason") or "see above"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
