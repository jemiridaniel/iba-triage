"""PWA launch experience: manifest, icons, iOS launch images, launch screen. No network."""

import json
import re
import struct
from pathlib import Path

import pytest

from tests.pipeline_fakes import REPO

PUBLIC = REPO / "frontend" / "public"
INDEX = (REPO / "frontend" / "index.html").read_text(encoding="utf-8")
MANIFEST = json.loads((PUBLIC / "manifest.webmanifest").read_text(encoding="utf-8"))
BRAND_TEAL = "#115e59"


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return struct.unpack(">II", header[16:24])


# --- manifest -------------------------------------------------------------------


def test_manifest_required_fields() -> None:
    assert MANIFEST["name"] == "Ibà — Fever Triage"
    assert MANIFEST["short_name"] == "Ibà"
    assert MANIFEST["display"] == "standalone"
    assert MANIFEST["start_url"] == MANIFEST["scope"] == "/"
    assert MANIFEST["background_color"] == MANIFEST["theme_color"] == BRAND_TEAL
    assert MANIFEST["description"] and MANIFEST["lang"]


def test_manifest_icons_exist_and_match_their_declared_sizes() -> None:
    icons = MANIFEST["icons"]
    assert {i["sizes"] for i in icons} >= {"192x192", "512x512"}
    assert any(i.get("purpose") == "maskable" and i["sizes"] == "512x512" for i in icons)
    for icon in icons:
        path = PUBLIC / icon["src"].lstrip("/")
        assert path.exists(), f"missing {path}"
        assert icon["type"] == "image/png"
        w, h = png_size(path)
        assert f"{w}x{h}" == icon["sizes"]


def test_apple_touch_icon_is_180() -> None:
    assert png_size(PUBLIC / "apple-touch-icon.png") == (180, 180)


def test_icon_has_no_cross_or_medical_emblem() -> None:
    # The red cross is a protected emblem; the mark is a plain lowercase "i".
    svg = (PUBLIC / "icon.svg").read_text(encoding="utf-8")
    assert "cross" not in svg.lower()
    assert svg.count("<rect") == 2 and svg.count("<circle") == 1  # background, stem, dot
    assert BRAND_TEAL in svg


# --- index.html: head -------------------------------------------------------------


def test_head_links_manifest_icons_and_ios_meta() -> None:
    assert '<link rel="manifest" href="/manifest.webmanifest" />' in INDEX
    assert '<link rel="apple-touch-icon" href="/apple-touch-icon.png" />' in INDEX
    assert '<meta name="apple-mobile-web-app-capable" content="yes" />' in INDEX
    assert 'name="apple-mobile-web-app-status-bar-style"' in INDEX
    assert 'name="apple-mobile-web-app-title" content="Ibà"' in INDEX
    assert f'<meta name="theme-color" content="{BRAND_TEAL}" />' in INDEX


@pytest.mark.parametrize(
    "link", re.findall(r'<link rel="apple-touch-startup-image"[^>]+>', INDEX, re.DOTALL)
)
def test_startup_images_exist_and_match_their_media_query(link: str) -> None:
    href = re.search(r'href="([^"]+)"', link).group(1)
    path = PUBLIC / href.lstrip("/")
    assert path.exists(), f"missing {path}"
    width, height = png_size(path)
    css_w = int(re.search(r"device-width: (\d+)px", link).group(1))
    css_h = int(re.search(r"device-height: (\d+)px", link).group(1))
    dpr = int(re.search(r"-webkit-device-pixel-ratio: (\d+)", link).group(1))
    assert (width, height) == (css_w * dpr, css_h * dpr)
    assert "orientation: portrait" in link


def test_startup_images_cover_current_iphones() -> None:
    links = re.findall(r'<link rel="apple-touch-startup-image"[^>]+>', INDEX, re.DOTALL)
    assert len(links) >= 6


# --- index.html: launch screen -----------------------------------------------------


def test_launch_screen_is_inline_and_offline_safe() -> None:
    assert 'id="launch"' in INDEX
    assert "Fever triage for primary health workers" in INDEX
    assert ">Ibà<" in INDEX
    style = re.search(r"<style>(.*?)</style>", INDEX, re.DOTALL).group(1)
    assert "#launch" in style and BRAND_TEAL in style
    assert "url(" not in style  # no external images
    assert "@font-face" not in style and "fonts.googleapis" not in INDEX
    assert "<svg" in INDEX.split('id="launch"')[1]  # the mark is inline, not a file


def test_launch_screen_handles_dark_mode_and_reduced_motion() -> None:
    style = re.search(r"<style>(.*?)</style>", INDEX, re.DOTALL).group(1)
    assert "prefers-color-scheme: dark" in style
    assert "prefers-reduced-motion: reduce" in style
    fade = re.search(r"transition: opacity (\d+)ms", style)
    assert 200 <= int(fade.group(1)) <= 300


def test_launch_screen_has_a_safety_timeout_and_no_minimum_duration() -> None:
    assert "__ibaLaunchTimer" in INDEX
    assert re.search(r"setTimeout\(function \(\) \{[^}]+\}, 4000\)", INDEX, re.DOTALL)
    main = (REPO / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
    assert "dismissLaunchScreen()" in main
    assert "requestAnimationFrame" in main  # dismissed on mount, not on a timer
    assert "setTimeout" not in main  # no artificial minimum


# --- header select (iOS zoom guard) -------------------------------------------------


def test_state_select_font_is_at_least_16px() -> None:
    app = (REPO / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    select = app.split("<select", 1)[1].split("</select>", 1)[0]
    assert "text-base" in select  # 16px: smaller makes iOS Safari zoom the page
    assert "min-h-[44px]" in select
    assert "focus-visible:ring-2" in select  # keyboard focus stays visible


# --- product name (Yoruba for fever) ------------------------------------------------

NAME = "Ibà"  # NFC, precomposed U+00E0


def test_manifest_and_title_use_the_accented_name() -> None:
    import unicodedata

    assert MANIFEST["short_name"] == NAME
    assert MANIFEST["name"].startswith(NAME)
    title = re.search(r"<title>(.*?)</title>", INDEX).group(1)
    assert title.startswith(NAME)
    for text in (MANIFEST["name"], MANIFEST["short_name"], title, INDEX):
        assert unicodedata.is_normalized("NFC", text)  # precomposed à, not a + U+0300


def test_launch_screen_and_icon_use_the_accented_name() -> None:
    assert f'<div class="iba-word">{NAME}</div>' in INDEX
    assert f'aria-label="{NAME} is starting"' in INDEX
    assert f'aria-label="{NAME}"' in (PUBLIC / "icon.svg").read_text(encoding="utf-8")


def test_referral_note_header_is_accented() -> None:
    nodes = (REPO / "backend" / "app" / "graph" / "nodes.py").read_text(encoding="utf-8")
    assert "IBÀ TRIAGE NOTE" in nodes


def test_about_explains_the_name() -> None:
    about = (REPO / "frontend" / "src" / "components" / "About.tsx").read_text(encoding="utf-8")
    assert "is the Yoruba word for fever" in about
    assert NAME in about
