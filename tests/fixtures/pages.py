"""
Fixture pages for the Level 0 tests (WEB_INSPECTION_PLAN.md section 16, Level 0).

Everything here is a mock page with a KNOWN expected result; no real site is ever contacted.
They are plain bytes in a dict so the tests, the local web server and the CLI test all use the
same source.
"""

from __future__ import annotations


def page(title: str = "", body: str = "", head: str = "") -> bytes:
    t = f"<title>{title}</title>" if title is not None else ""
    return f"<!DOCTYPE html><html><head><meta charset=utf-8>{t}{head}</head><body>{body}</body></html>".encode()


def anchors(hrefs) -> str:
    return "".join(f'<a href="{h}">link{i}</a>' for i, h in enumerate(hrefs))


# All pages below are judged as if served from this URL (registrable domain example.com).
BASE_URL = "https://www.example.com/shop/index.html"

PAGES = {
    # baseline
    "plain": page("Plain page", "<p>Hello, this is a perfectly ordinary page with enough text to "
                                "not be near-empty. " * 3 + "</p>"),
    # 10 links, all on the same registrable domain (relative, absolute, subdomain)
    "html_all_internal": page("All internal", anchors(
        ["/a", "b.html", "../c", "https://www.example.com/d", "https://shop.example.com/e",
         "http://example.com/f", "//cdn.example.com/g", "/h?x=1", "/i#frag", "/j"])),
    # 5 internal, 3 external, 2 null ('#' and javascript:void(0))
    "html_mixed_links": page("Mixed links", anchors(
        ["/a", "/b", "/c", "https://www.example.com/d", "/e",
         "https://other.org/x", "https://evil.net/y", "//third.io/z",
         "#", "javascript:void(0)"])),
    "html_ext_css": page("Ext css", "", head=(
        '<link rel="stylesheet" href="//cdn.other.net/a.css">'
        '<link rel="stylesheet" href="https://fonts.other.org/b.css">'
        '<link rel="stylesheet" href="/local.css">')),
    "html_login_ext_action": page("Login", '<form action="https://evil.net/post.php"><input type=password></form>'),
    "html_login_empty_action": page("Login", '<form action=""><input type=password></form>'),
    "html_login_hash_action": page("Login", '<form action="#"><input type=password></form>'),
    "html_login_blank_action": page("Login", '<form action="about:blank"><input type=password></form>'),
    "html_login_php_action": page("Login", '<form action="login.php"><input type=password></form>'),
    "html_login_normal_action": page("Contact", '<form action="/contact"><input name=q></form>'),
    "html_form_no_action": page("Contact", '<form><input name=q></form>'),
    "html_ext_favicon": page("Fav", "", head='<link rel="shortcut icon" href="https://other.com/favicon.ico">'),
    "html_int_favicon": page("Fav", "", head='<link rel="icon" href="/favicon.ico">'),
    "html_icon_in_body": page("Fav", '<link rel="icon" href="https://other.com/favicon.ico">'),
    # media: img internal x3, img external x1, video external x1, audio internal x1
    "html_media_mix": page("Media", (
        '<img src="/a.png"><img src="b.png"><img src="https://www.example.com/c.png">'
        '<img src="https://cdn.other.com/d.png"><video src="https://cdn.other.com/e.mp4"></video>'
        '<audio src="/f.mp3"></audio>')),
    "html_hidden_iframe": page("Frame", '<iframe src="https://x.com/a" width="0" height="0"></iframe>'),
    "html_display_none_iframe": page("Frame", '<iframe src="https://x.com/a" style="display:none"></iframe>'),
    "html_visible_iframe": page("Frame", '<iframe src="https://x.com/a" width="300" height="200"></iframe>'),
    "html_no_title": b"<html><head></head><body><p>no title here at all, but plenty of words to read.</p></body></html>",
    "html_empty_title": page("", "<p>empty title above, plenty of visible words in here.</p>"),
    "html_domain_in_title": page("Welcome to Example Shop", "<p>body body body body body body body body</p>"),
    "html_domain_not_in_title": page("Welcome to the shop", "<p>body body body body body body body body</p>"),
    "html_copyright_domain": page("T", "<footer>&copy; 2026 Example Inc. All rights reserved.</footer>"),
    "html_copyright_other": page("T", "<footer>&copy; 2026 Acme Corporation. All rights reserved.</footer>"),
    "html_no_copyright": page("T", "<footer>All rights reserved by nobody.</footer>"),
    "html_copyright_far": page("T", "<p>example</p>" + "<p>" + "x" * 120 + "</p>" +
                                    "<footer>&copy; 2026 Somebody Else</footer>"),
    # links in <meta>, <script>, <link>: 3 internal, 1 external -> 75.0
    "html_meta_script_link_links": page("Tags", "<script src='/a.js'></script>"
                                        "<script src='https://cdn.other.com/b.js'></script>",
                                        head="<link rel='stylesheet' href='/x.css'>"
                                             "<meta property='og:image' content='https://www.example.com/o.png'>"),
    # JavaScript-built shell: tiny, scripts, no visible text
    "html_empty": b"<html><head></head><body><div id=root></div><script>var a=1;</script><script>var b=2;</script></body></html>",
    # redirects found by pattern
    "meta_refresh": b"<html><head><meta http-equiv='refresh' content='0; url=/page/plain'></head><body></body></html>",
    "meta_refresh_slow": b"<html><head><meta http-equiv='refresh' content='60; url=/page/plain'></head><body>ok</body></html>",
    "meta_refresh_abs": b"<html><head><meta http-equiv=\"refresh\" content=\"3; URL='https://www.other.com/landing'\"></head></html>",
    "js_redirect": b"<html><body><script>window.location.href = '/page/plain';</script></body></html>",
    "js_replace": b"<html><body><script>location.replace(\"https://www.other.com/x\");</script></body></html>",
    "js_assign": b"<html><body><script>document.location = 'https://www.other.com/y'</script></body></html>",
    # verdicts
    "challenge": b"<html><head><title>Just a moment...</title></head><body><p>Checking your browser before accessing the site.</p></body></html>",
    "challenge_body": b"<html><head><title>x</title></head><body><div class='cf-browser-verification'>Please wait</div></body></html>",
    "parked": b"<html><head><title>example.com</title></head><body><h1>This domain is for sale</h1><p>Buy this domain today.</p></body></html>",
    "default_nginx": b"<html><head><title>Welcome to nginx!</title></head><body><h1>Welcome to nginx!</h1></body></html>",
    "directory_listing": b"<html><head><title>Index of /</title></head><body><h1>Index of /</h1></body></html>",
    # hostile content
    "evil_strings": ("<html><head><title>=cmd|' /C calc'!A0 \x1b[31mred\x1b[0m ‮evil‬ " + "T" * 10000 +
                     "</title></head><body><p>x</p></body></html>").encode("utf-8"),
    "latin1": "<html><head><meta charset='iso-8859-1'><title>Caf\xe9 Example</title></head><body><p>caf\xe9 ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok</p></body></html>".encode("latin-1"),
}


def malformed(depth: int = 3000) -> bytes:
    """Broken tags and very deep nesting."""
    return ("<html><body>" + "<div>" * depth + "<a href='/x'>unclosed<b><i></a></b>" + "</p></span>" * 50).encode()


def big_html(size: int) -> bytes:
    """A valid page of roughly `size` bytes."""
    chunk = "<p>filler text for a big page. lorem ipsum dolor sit amet.</p>\n"
    n = max(1, size // len(chunk))
    return ("<html><head><title>Big</title></head><body>" + chunk * n + "</body></html>").encode()
