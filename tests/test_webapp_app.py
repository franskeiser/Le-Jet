"""
Tests for webapp/app.py with a fake lab pipeline (no network). Skipped where fastapi is not installed
(the website's own requirements: webapp/requirements-web.txt).

Run with:  python -m pytest tests/test_webapp_app.py
"""

from __future__ import annotations

import asyncio
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient    # noqa: E402

from lejet import schema                      # noqa: E402
from webapp import app as webapp_app          # noqa: E402

TEXT = "Your GCash account is locked. Verify at https://gcash-verify-ph.xyz/login now."
CODE = {"X-Access-Code": "s3cret"}


class _Insp:
    def __init__(self, row):
        self.row = row


class FakePipeline:
    rdap = None
    calls = []
    dead = False                 # True: the page is not reached (no html, no HTTP answer)

    def __init__(self, settings=None):
        pass

    async def start(self):
        return self

    async def close(self):
        pass

    async def inspect(self, url_id, raw_url, batch_id):
        FakePipeline.calls.append(raw_url)
        row = schema.new_row(url_id, raw_url, batch_id)
        row.update(fail_reason="none", html_available=1, page_fetched=1, final_status=200, final_url=raw_url,
                   reg_domain="gcash-verify-ph.xyz", final_reg_domain="gcash-verify-ph.xyz", nb_hyperlinks=7)
        if FakePipeline.dead:
            row.update(fail_reason="dns_nxdomain", html_available=0, page_fetched=0, final_status=-1, final_url="", nb_hyperlinks=-1)
        return _Insp(row)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp_app, "Pipeline", FakePipeline)
    monkeypatch.setattr(webapp_app, "ACCESS_CODE", "s3cret")
    monkeypatch.setattr(webapp_app, "OPEN", False)
    monkeypatch.setattr(webapp_app, "DEV", False)
    monkeypatch.setattr(webapp_app, "DATASET_PATH", str(tmp_path / "s.csv"))
    FakePipeline.calls = []
    with TestClient(webapp_app.app) as c:
        yield c


@pytest.fixture()
def open_client(tmp_path, monkeypatch):
    """The deployed setup: LEJET_OPEN=1 and no access code."""
    monkeypatch.setattr(webapp_app, "Pipeline", FakePipeline)
    monkeypatch.setattr(webapp_app, "ACCESS_CODE", "")
    monkeypatch.setattr(webapp_app, "OPEN", True)
    monkeypatch.setattr(webapp_app, "DEV", False)
    monkeypatch.setattr(webapp_app, "DATASET_PATH", str(tmp_path / "s.csv"))
    FakePipeline.calls = []
    with TestClient(webapp_app.app) as c:
        yield c


def test_open_mode_needs_no_code(open_client):
    assert open_client.get("/api/health").json()["code_required"] is False
    r = _submit(open_client, headers={})
    assert r.status_code == 200 and r.json()["message_id"] == "web_000001"
    assert open_client.get("/api/rows").json()["total"] == 1
    assert open_client.get("/api/dataset.csv").status_code == 200
    assert open_client.get("/robots.txt").text.startswith("User-agent: *")
    assert open_client.get("/").headers["x-robots-tag"].startswith("noindex")


def test_server_refuses_to_start_unprotected_by_accident(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp_app, "Pipeline", FakePipeline)
    monkeypatch.setattr(webapp_app, "ACCESS_CODE", "")
    monkeypatch.setattr(webapp_app, "OPEN", False)
    monkeypatch.setattr(webapp_app, "DEV", False)
    monkeypatch.setattr(webapp_app, "DATASET_PATH", str(tmp_path / "s.csv"))
    with pytest.raises(RuntimeError):
        with TestClient(webapp_app.app):
            pass


def _submit(client, text=TEXT, label="smishing", language="en", headers=CODE, **extra):
    body = {"sms_text": text, "label": label, "language": language, **extra}
    return client.post("/api/submit", json={k: v for k, v in body.items() if v is not None}, headers=headers)


def test_every_api_call_needs_the_code(client):
    assert client.get("/api/rows").status_code == 401
    assert client.get("/api/dataset.csv", headers={"X-Access-Code": "nope"}).status_code == 401
    assert _submit(client, headers={}).status_code == 401
    assert client.get("/api/health").status_code == 200


def test_preview_finds_the_link_without_saving(client):
    d = client.post("/api/preview", json={"sms_text": TEXT}, headers=CODE).json()
    assert d["link"].startswith("hxxps://gcash-verify-ph[.]xyz") and d["links_in_message"] == 1
    assert client.post("/api/preview", json={"sms_text": "hello"}, headers=CODE).json()["link"] is None
    assert client.get("/api/rows", headers=CODE).json()["total"] == 0 and FakePipeline.calls == []


def test_submit_inspects_the_found_link_and_appends_a_row(client):
    r = _submit(client)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["message_id"] == "web_000001" and d["fetch_status"] == "live" and d["total"] == 1
    assert d["link"].startswith("hxxps://") and d["features"]["nb_hyperlinks"] == 7 and len(d["features"]) == 37
    assert FakePipeline.calls == ["https://gcash-verify-ph.xyz/login"]        # the link, taken from the text

    rows = client.get("/api/rows", headers=CODE).json()
    assert rows["counts"]["smishing"] == 1 and rows["rows"][0]["message_id"] == "web_000001"
    csv_text = client.get("/api/dataset.csv", headers=CODE).text
    assert csv_text.splitlines()[0].startswith("message_id,sms_text,label") and csv_text.count("web_000001") == 1


def test_the_csv_has_the_50_columns_and_the_chosen_language(client):
    import csv, io
    for text, language in ((TEXT, "taglish"), ("Hi po! Claim your prize https://x-prize.top/a", "tl"), ("Win now https://y-prize.top/b", "en")):
        assert _submit(client, text=text, language=language).status_code == 200
    rows = list(csv.DictReader(io.StringIO(client.get("/api/dataset.csv", headers=CODE).text)))
    assert [r["language"] for r in rows] == ["taglish", "tl", "en"]
    assert len(rows[0]) == 50
    for gone in ("category", "sender_type", "date_received", "label_source", "consent", "inspected_at", "hours_to_inspect"):
        assert gone not in rows[0]
    got = client.get("/api/rows", headers=CODE).json()
    assert got["languages"] == {"en": 1, "tl": 1, "taglish": 1} and got["rows"][0]["language"] == "en"


def test_a_client_that_still_sends_consent_is_not_broken(client):
    assert _submit(client, consent=True).status_code == 200                      # an old page: the field is simply ignored


def test_rejections_save_nothing(client):
    assert _submit(client, text="no link in this one").status_code == 422
    assert _submit(client, label="maybe").status_code == 422
    assert _submit(client, language="english").status_code == 422                # only en / tl / taglish
    assert _submit(client, language="-1").status_code == 422
    assert _submit(client, language=None).status_code == 422                     # the language is required, like the label
    assert _submit(client, text="   ").status_code in (400, 422)
    assert _submit(client).status_code == 200
    again = _submit(client)
    assert again.status_code == 409 and "web_000001" in again.json()["detail"]
    assert client.get("/api/rows", headers=CODE).json()["total"] == 1
    assert len(FakePipeline.calls) == 1                                       # the duplicate was never fetched


# ---------------------------------------------------------------------------------------------- screenshots
from webapp import screens    # noqa: E402


class FakeReader:
    def __init__(self):
        pass


@pytest.fixture(autouse=True)
def fake_ocr(monkeypatch):
    """No OCR models in these tests: the reader is a stand-in and 'reading' returns fixed bubbles."""
    monkeypatch.setattr(screens, "Reader", FakeReader)

    def fake_read(reader, data):
        if data.startswith(b"BAD"):
            raise screens.ImageRejected("This file is not a picture.")
        if data.startswith(b"EMPTY"):
            return []
        return [screens.Bubble("GCash: locked. Verify at https://gcash-verify-ph.xyz/login", "in", 80, 400, 700, 500),
                screens.Bubble("who is this?", "out", 700, 600, 1000, 640)]

    monkeypatch.setattr(screens, "read_screenshot", fake_read)


def _upload(client, *blobs, headers=CODE):
    files = [("files", (f"shot{i}.png", b, "image/png")) for i, b in enumerate(blobs)]
    return client.post("/api/screenshots", files=files, headers=headers)


def test_screenshots_return_messages_and_save_nothing(client):
    r = _upload(client, b"PNGDATA")
    assert r.status_code == 200, r.text
    msgs = r.json()["images"][0]["messages"]
    assert [m["direction"] for m in msgs] == ["in", "out"]
    assert msgs[0]["link"].startswith("hxxps://gcash-verify-ph[.]xyz") and msgs[1]["link"] is None
    assert client.get("/api/rows", headers=CODE).json()["total"] == 0 and FakePipeline.calls == []
    assert client.get("/api/health").json()["screenshots"] is True


def test_screenshot_problems_are_reported_per_picture(client):
    imgs = _upload(client, b"BAD file", b"EMPTY", b"PNGDATA").json()["images"]
    assert imgs[0]["error"] == "This file is not a picture." and imgs[0]["messages"] == []
    assert imgs[1]["error"] == "No text was found in this picture."
    assert imgs[2]["error"] is None and len(imgs[2]["messages"]) == 2
    big = _upload(client, b"x" * (screens.MAX_IMAGE_BYTES + 5)).json()["images"][0]
    assert "Larger than" in big["error"]


def test_screenshot_limits_and_access(client):
    assert _upload(client, *([b"PNGDATA"] * (screens.MAX_IMAGES + 1))).status_code == 400
    assert _upload(client, b"PNGDATA", headers={}).status_code == 401
    assert _upload(client, *([b"PNGDATA"] * 6)).status_code == 200
    assert _upload(client, *([b"PNGDATA"] * 6)).status_code == 200
    assert _upload(client, b"PNGDATA").status_code == 429                      # 12 pictures a minute per person


def test_screenshots_are_off_when_the_ocr_cannot_load(client, monkeypatch, tmp_path):
    def broken():
        raise ImportError("no rapidocr")

    monkeypatch.setattr(screens, "Reader", broken)
    monkeypatch.setattr(webapp_app, "DATASET_PATH", str(tmp_path / "other.csv"))
    with TestClient(webapp_app.app) as c:
        assert c.get("/api/health").json()["screenshots"] is False
        assert _upload(c, b"PNGDATA").status_code == 503
        assert c.post("/api/preview", json={"sms_text": "see https://a.example.com"}, headers=CODE).status_code == 200   # pasting still works


# ---------------------------------------------------------------------------------------------- automatic page_rank
class FakeRanker:
    """Stands in for webapp.rank.RankClient: answers from a table, records what it was asked."""
    enabled = True
    last_error = ""
    asked: list = []
    answers = {"gcash-verify-ph.xyz": 7}

    @classmethod
    def from_file(cls, *args, **kwargs):
        return cls()

    async def lookup(self, domain):
        FakeRanker.asked.append(domain)
        return FakeRanker.answers.get(domain, -1)

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def fake_rank(monkeypatch):
    monkeypatch.setattr(webapp_app.rank, "RankClient", FakeRanker)
    monkeypatch.setattr(FakeRanker, "asked", [])
    monkeypatch.setattr(FakeRanker, "enabled", True)
    monkeypatch.setattr(FakePipeline, "dead", False)


def _csv_rows(client):
    import csv, io
    return list(csv.DictReader(io.StringIO(client.get("/api/dataset.csv", headers=CODE).text)))


def test_page_rank_is_filled_in_automatically_for_a_page_that_loaded(client):
    d = _submit(client).json()
    assert d["features"]["page_rank"] == 7
    assert FakeRanker.asked == ["gcash-verify-ph.xyz"]                       # the final registrable domain
    assert _csv_rows(client)[0]["page_rank"] == "7"


def test_a_page_that_was_not_reached_is_not_looked_up(client, monkeypatch):
    monkeypatch.setattr(FakePipeline, "dead", True)
    d = _submit(client).json()
    assert d["fetch_status"] == "dead" and d["features"]["page_rank"] == -1
    assert FakeRanker.asked == []                                            # its domain is not sent to the service
    assert _csv_rows(client)[0]["page_rank"] == "-1"


def test_a_failed_lookup_still_saves_the_row_with_minus_one(client, monkeypatch):
    monkeypatch.setattr(FakeRanker, "answers", {})                           # the service gives nothing: -1
    assert _submit(client).status_code == 200
    assert _csv_rows(client)[0]["page_rank"] == "-1"


def test_a_ranker_that_hangs_does_not_hang_the_submission(client, monkeypatch):
    async def slow(self, domain):
        await asyncio.sleep(60)

    monkeypatch.setattr(FakeRanker, "lookup", slow)
    real_wait_for = asyncio.wait_for

    async def quick_wait_for(aw, timeout):                                   # the 15 s limit, shortened for the test
        return await real_wait_for(aw, 0.2)

    monkeypatch.setattr(webapp_app.asyncio, "wait_for", quick_wait_for)
    assert _submit(client).status_code == 200
    assert _csv_rows(client)[0]["page_rank"] == "-1"


def test_health_shows_why_page_rank_failed(client, monkeypatch):
    assert client.get("/api/health").json()["page_rank_error"] is None
    monkeypatch.setattr(FakeRanker, "last_error", "HTTP 429")
    with TestClient(webapp_app.app) as c2:
        assert c2.get("/api/health").json()["page_rank_error"] == "HTTP 429"


def test_health_says_whether_page_rank_is_on(client, monkeypatch):
    assert client.get("/api/health").json()["page_rank"] is True
    monkeypatch.setattr(FakeRanker, "enabled", False)
    with TestClient(webapp_app.app) as c2:
        assert c2.get("/api/health").json()["page_rank"] is False
