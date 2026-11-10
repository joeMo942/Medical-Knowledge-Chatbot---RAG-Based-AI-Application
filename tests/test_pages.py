"""The two static pages (chat and admin): no markup built from dynamic text, nothing loaded from outside."""
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "medrag" / "static"
PAGES = {name: (STATIC / name).read_text(encoding="utf-8") for name in ("index.html", "admin.html")}


@pytest.mark.parametrize("name", PAGES)
def test_page_never_builds_markup_from_dynamic_text(name):
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in PAGES[name], banned


@pytest.mark.parametrize("name", PAGES)
def test_page_loads_nothing_from_outside(name):
    page = PAGES[name]
    assert not re.search(r"""(src|href|action)\s*=\s*["']?(https?:)?//""", page, re.I)
    assert not re.search(r"""@import|url\(\s*["']?(https?:)?//""", page, re.I)
    assert "<script src" not in page and "<link" not in page


@pytest.mark.parametrize("name", PAGES)
def test_page_is_themed_and_fits_a_phone(name):
    page = PAGES[name]
    assert "prefers-color-scheme: dark" in page and 'name="viewport"' in page
    assert "Noto Sans Arabic" in page


def test_admin_page_sends_the_bearer_token_and_keeps_it_in_session_storage_only():
    page = PAGES["admin.html"]
    assert '"Authorization": "Bearer " + token' in page and "sessionStorage" in page and "localStorage" not in page
    for path in ("/admin/documents", "/admin/analytics", "/admin/handoffs"):
        assert path in page
