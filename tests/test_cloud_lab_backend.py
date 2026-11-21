"""
Tests for the pieces that connect the tool to the Azure cloud lab: which lab is chosen, and how the extra
(linked-page) screenshots are stored and served. No network, no VM.

Run with:  python -m pytest tests/test_cloud_lab_backend.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.serving import live_inspect as li
from src.serving.api import EVIDENCE_RE
from src.serving.lejet_service import LinkStore
from test_pngcheck import png

A = "0123456789abcdef"


def test_the_environment_chooses_the_lab(monkeypatch):
    monkeypatch.setenv("LEJET_LAB", "azure")
    assert li.lab_backend() == "azure"
    monkeypatch.setenv("LEJET_LAB", "LOCAL")
    assert li.lab_backend() == "local"


def test_without_a_choice_the_ssh_alias_decides(monkeypatch, tmp_path):
    monkeypatch.delenv("LEJET_LAB", raising=False)
    (tmp_path / ".ssh").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert li.lab_backend() == "local"                                       # no ssh config at all
    (tmp_path / ".ssh" / "config").write_text("Host other\n  HostName x\n")
    assert li.lab_backend() == "local"
    (tmp_path / ".ssh" / "config").write_text(f"Host {li.AZURE_HOST}\n  HostName x\n  User u\n")
    assert li.lab_backend() == "azure"


@pytest.mark.parametrize("path, want", [
    (f"/evidence/{A}.png", (A, None)),
    (f"/evidence/{A}_p1.png", (A, "1")),
    (f"/evidence/{A}_p4.png", (A, "4")),
])
def test_evidence_urls_for_the_landing_page_and_linked_pages(path, want):
    assert EVIDENCE_RE.match(path).groups() == want


@pytest.mark.parametrize("path", [f"/evidence/{A}_p0.png", f"/evidence/{A}_p5.png", f"/evidence/{A}_p10.png",
                                  f"/evidence/{A}_px.png", f"/evidence/../{A}.png", f"/evidence/{A}_p1.png.exe"])
def test_other_evidence_urls_are_refused(path):
    assert EVIDENCE_RE.match(path) is None


def test_the_store_serves_the_landing_page_and_its_linked_pages(tmp_path):
    shots = tmp_path / "batch"
    shots.mkdir()
    (shots / f"{A}.png").write_bytes(png(8, 8))
    (shots / f"{A}_p1.png").write_bytes(png(9, 8))
    (shots / f"{A}_p3.png").write_bytes(png(10, 8))
    (shots / f"{A}_p9.png").write_bytes(png(11, 8))                           # not a valid page number: ignored
    store = LinkStore(csv=tmp_path / "none.csv", shot_dirs=[tmp_path])
    assert store.shot_path(A).name == f"{A}.png"
    assert store.shot_path(A, 1).name == f"{A}_p1.png"
    assert store.shot_path(A, 2) is None
    assert store.shot_path(A, 3).name == f"{A}_p3.png"
    assert store.page_count(A) == 2
    assert store.page_count("not-an-id") == 0 and store.shot_path("not-an-id", 1) is None


# ---------------------------------------------------------------------------------------
# archived copies (host side)
# ---------------------------------------------------------------------------------------
import json  # noqa: E402

from src.serving import archive_fallback as af  # noqa: E402

UUID = "0123abcd-0123-4abc-8def-0123456789ab"


def fake_fetch(table):
    def fetch(url, headers, limit):
        for prefix, value in table.items():
            if url.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return value if isinstance(value, bytes) else json.dumps(value).encode()
        raise OSError("no such fake url")
    return fetch


WAYBACK_OK = {"archived_snapshots": {"closest": {"available": True, "timestamp": "20260101120000",
                                                  "url": "http://web.archive.org/web/20260101120000/http://gone.example/"}}}


def test_the_fallback_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("LEJET_ARCHIVE_FALLBACK", raising=False)
    assert af.find("http://gone.example/", fake_fetch({"https://archive.org": WAYBACK_OK})) is None


def test_wayback_gives_a_snapshot_link_on_the_expected_host_only():
    got = af.wayback("http://gone.example/", fake_fetch({"https://archive.org": WAYBACK_OK}))
    assert got["source"] == "archived_wayback" and got["url"].startswith("https://web.archive.org/web/2026")
    evil = json.loads(json.dumps(WAYBACK_OK))
    evil["archived_snapshots"]["closest"]["url"] = "http://evil.example/web/2026/x"
    assert af.wayback("http://gone.example/", fake_fetch({"https://archive.org": evil})) is None
    assert af.wayback("http://gone.example/", fake_fetch({"https://archive.org": {"archived_snapshots": {}}})) is None
    assert af.wayback("http://gone.example/", fake_fetch({"https://archive.org": OSError("down")})) is None


def test_urlscan_needs_a_key_and_returns_only_a_valid_png():
    table = {"https://urlscan.io/api/v1/search/": {"results": [{"_id": UUID, "task": {"time": "2026-01-01T00:00:00Z"}}]},
             "https://urlscan.io/screenshots/": png(8, 8)}
    assert af.urlscan("http://gone.example/", fake_fetch(table), key="") is None
    got = af.urlscan("http://gone.example/", fake_fetch(table), key="k")
    assert got["source"] == "archived_urlscan" and got["image"] == png(8, 8) and UUID in got["url"]
    table["https://urlscan.io/screenshots/"] = b"<html>not a png</html>"
    assert af.urlscan("http://gone.example/", fake_fetch(table), key="k") is None
    table["https://urlscan.io/api/v1/search/"] = {"results": [{"_id": "../../etc/passwd"}]}
    assert af.urlscan("http://gone.example/", fake_fetch(table), key="k") is None


def test_find_prefers_the_stored_screenshot_then_falls_back_to_wayback(monkeypatch):
    monkeypatch.setenv("LEJET_ARCHIVE_FALLBACK", "1")
    table = {"https://archive.org": WAYBACK_OK}
    assert af.find("http://gone.example/", fake_fetch(table), key="k")["source"] == "archived_wayback"
    table["https://urlscan.io/api/v1/search/"] = {"results": [{"_id": UUID, "task": {}}]}
    table["https://urlscan.io/screenshots/"] = png(8, 8)
    assert af.find("http://gone.example/", fake_fetch(table), key="k")["source"] == "archived_urlscan"


# ---------------------------------------------------------------------------------------
# which picture a link shows: its own check's picture, or one clearly marked as EARLIER
# ---------------------------------------------------------------------------------------
import os  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402


def make_store(tmp_path, batch_of_row, folders):
    """folders: {batch folder name: {file name: age in days}}. One stored row for link A in `batch_of_row`."""
    root = tmp_path / "shots"
    for folder, files in folders.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        for name, age_days in files.items():
            f = root / folder / name
            f.write_bytes(png(8, 8))
            t = time.time() - age_days * 86400
            os.utime(f, (t, t))
    csv = tmp_path / "rows.csv"
    pd.DataFrame([{"link_id": A, "key": "k1", "batch_id": batch_of_row}]).to_csv(csv, index=False)
    return LinkStore(csv=tmp_path / "none.csv", shot_dirs=[root], on_demand_csv=csv), root


def test_the_picture_of_the_rows_own_check_wins_over_an_older_one(tmp_path):
    # "a_old" sorts first and used to win; the row describes batch b_new
    store, _ = make_store(tmp_path, "b_new", {"a_old": {f"{A}.png": 30}, "b_new": {f"{A}.png": 0}})
    assert store.shot_path(A).parent.name == "b_new"
    info = store.shot_info(A)
    assert info["has_screenshot"] and info["screenshot_current"] and info["screenshot_taken"] == ""


def test_a_failed_new_check_shows_the_old_picture_labelled_earlier_with_its_date(tmp_path):
    # the row's batch has NO picture (the lab could not open the page); an older batch has one from 30 days ago
    store, _ = make_store(tmp_path, "b_new", {"a_old": {f"{A}.png": 30}, "b_new": {}})
    info = store.shot_info(A)
    assert info["has_screenshot"] is True and info["screenshot_current"] is False
    assert info["screenshot_taken"] == time.strftime("%Y-%m-%d", time.localtime(time.time() - 30 * 86400))
    assert store.shot_path(A).parent.name == "a_old"                          # still servable, but flagged


def test_with_several_earlier_pictures_the_newest_is_the_one_shown(tmp_path):
    store, _ = make_store(tmp_path, "b_new", {"x1": {f"{A}.png": 50}, "x2": {f"{A}.png": 5}, "b_new": {}})
    assert store.shot_path(A).parent.name == "x2" and store.shot_info(A)["screenshot_current"] is False


def test_linked_pages_always_come_from_the_same_check_as_the_landing_page(tmp_path):
    store, _ = make_store(tmp_path, "b_new", {
        "a_old": {f"{A}.png": 30, f"{A}_p1.png": 30, f"{A}_p2.png": 30},
        "b_new": {f"{A}.png": 0, f"{A}_p1.png": 0}})
    assert store.page_count(A) == 1 and store.shot_path(A, 1).parent.name == "b_new" and store.shot_path(A, 2) is None
    # and when only the old check has pictures, its pages come along (as a set), still marked earlier
    store2, _ = make_store(tmp_path / "second", "b_new", {"a_old": {f"{A}.png": 30, f"{A}_p1.png": 30, f"{A}_p2.png": 30}, "b_new": {}})
    assert store2.page_count(A) == 2 and store2.shot_info(A)["screenshot_current"] is False


def test_a_fresh_inspection_switches_the_link_from_earlier_to_current(tmp_path):
    store, root = make_store(tmp_path, "b_old_row", {"a_old": {f"{A}.png": 30}})
    assert store.shot_info(A)["screenshot_current"] is False                  # row batch "b_old_row" has no picture
    new = root / "b_fresh"
    new.mkdir()
    (new / f"{A}.png").write_bytes(png(9, 9))
    record = {"link_id": A, "key": "k1", "batch_id": "b_fresh"}
    store.add(record, tmp_path / "on_demand.csv", new)
    info = store.shot_info(A)
    assert info["screenshot_current"] is True and store.shot_path(A).parent.name == "b_fresh"


def test_a_link_with_no_picture_at_all_says_so(tmp_path):
    store, _ = make_store(tmp_path, "b_new", {"b_new": {}})
    assert store.shot_info(A) == {"has_screenshot": False, "screenshot_current": False, "screenshot_taken": "",
                                  "screenshot_pages": 0, "screenshot_items": []}
    assert store.shot_path(A) is None and not store.has_shot(A)


def test_each_picture_comes_with_its_size_title_and_address(tmp_path):
    folder = tmp_path / "shots" / "b1"
    folder.mkdir(parents=True)
    (folder / f"{A}.png").write_bytes(png(12, 40))
    (folder / f"{A}_p1.png").write_bytes(png(12, 20))
    manifest = {"url_id": A, "ok": True, "final_url": "hxxps://shop[.]example/", "title": "Shop home", "truncated": False,
                "pages": [{"n": 1, "url": "hxxps://shop[.]example/login", "title": "Sign in", "width": 12, "height": 20,
                           "bytes": 1, "sha256": "a" * 64, "text_chars": 3, "truncated": True}]}
    (folder / "shots.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    csv = tmp_path / "rows.csv"
    pd.DataFrame([{"link_id": A, "key": "k1", "batch_id": "b1"}]).to_csv(csv, index=False)
    store = LinkStore(csv=tmp_path / "none.csv", shot_dirs=[tmp_path / "shots"], on_demand_csv=csv)
    items = store.shot_info(A)["screenshot_items"]
    assert [i["n"] for i in items] == [0, 1]
    assert (items[0]["width"], items[0]["height"], items[0]["title"], items[0]["url"]) == (12, 40, "Shop home", "hxxps://shop[.]example/")
    assert (items[1]["height"], items[1]["title"], items[1]["url"], items[1]["truncated"]) == (20, "Sign in", "hxxps://shop[.]example/login", True)


def test_older_batches_without_a_manifest_still_list_their_pictures_with_sizes(tmp_path):
    store, _ = make_store(tmp_path, "b1", {"b1": {f"{A}.png": 0, f"{A}_p2.png": 0}})
    items = store.shot_info(A)["screenshot_items"]
    assert [i["n"] for i in items] == [0, 2] and all(i["title"] == "" and i["url"] == "" for i in items)
    assert all(i["width"] == 8 and i["height"] == 8 for i in items)


def test_text_from_the_manifest_is_cleaned_and_capped(tmp_path):
    folder = tmp_path / "shots" / "b1"
    folder.mkdir(parents=True)
    (folder / f"{A}.png").write_bytes(png(8, 8))
    bad = {"url_id": A, "final_url": "hxxp://x[.]y/" + "a" * 500, "title": "T" + chr(0) + "i" + chr(7) + "tle" + "z" * 300, "pages": []}
    (folder / "shots.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    csv = tmp_path / "rows.csv"
    pd.DataFrame([{"link_id": A, "key": "k1", "batch_id": "b1"}]).to_csv(csv, index=False)
    store = LinkStore(csv=tmp_path / "none.csv", shot_dirs=[tmp_path / "shots"], on_demand_csv=csv)
    item = store.shot_info(A)["screenshot_items"][0]
    assert item["title"].startswith("Title") and len(item["title"]) <= 120 and len(item["url"]) <= 300
    assert not any(ord(c) < 32 for c in item["title"])
