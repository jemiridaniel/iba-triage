"""Generate the Ibà app icons and iOS launch images from one vector definition.

    uv run python -m scripts.make_icons

Writes frontend/public/: icon.svg, icon-192.png, icon-512.png, icon-512-maskable.png,
apple-touch-icon.png (180) and splash/*.png. Pure Python (no image libraries): the mark is
simple geometry, drawn with 3x3 supersampling and written as PNG with zlib.

Mark: a lowercase "i" (dot + stem) in white on the brand teal. Deliberately not a
thermometer, and never a cross: the red cross is a protected emblem.
"""

import struct
import sys
import zlib
from pathlib import Path

PUBLIC = Path("frontend/public")

TEAL = (0x11, 0x5E, 0x59)  # brand: tailwind teal-800, same as the app header
WHITE = (0xFF, 0xFF, 0xFF)
SS = 3  # supersampling factor per axis

# Geometry as fractions of the icon size.
DOT_CY, DOT_R = 0.16, 0.115
STEM_TOP, STEM_BOTTOM, STEM_HALF_W = 0.37, 0.88, 0.115
CORNER = 0.22  # rounded-square radius (ignored for full-bleed/maskable icons)


def _round_rect(x: float, y: float, x0: float, y0: float, x1: float, y1: float, r: float) -> bool:
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return False
    cx = min(max(x, x0 + r), x1 - r)
    cy = min(max(y, y0 + r), y1 - r)
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _mark(x: float, y: float, size: float, scale: float, cx: float, cy: float) -> bool:
    """True if (x, y) is inside the "i", drawn at `scale` of `size`, centred on (cx, cy)."""
    u = (x - cx) / (size * scale) + 0.5
    v = (y - cy) / (size * scale) + 0.5
    if (u - 0.5) ** 2 + (v - DOT_CY) ** 2 <= DOT_R**2:
        return True
    return _round_rect(
        u, v, 0.5 - STEM_HALF_W, STEM_TOP, 0.5 + STEM_HALF_W, STEM_BOTTOM, STEM_HALF_W
    )


def render(
    w: int,
    h: int,
    *,
    mark_scale: float,
    rounded: bool = False,
    bg: tuple[int, int, int] = TEAL,
) -> bytes:
    """RGBA rows for a `bg` canvas with the white mark centred."""
    size = min(w, h)
    cx, cy = w / 2, h / 2
    radius = size * CORNER
    rows = bytearray()
    for py in range(h):
        row = bytearray([0])
        for px in range(w):
            inside = cover = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) / SS
                    y = py + (sy + 0.5) / SS
                    on_canvas = _round_rect(x, y, 0, 0, w, h, radius) if rounded else True
                    if on_canvas:
                        cover += 1
                        if _mark(x, y, size, mark_scale, cx, cy):
                            inside += 1
            n = SS * SS
            if not cover:
                row += bytes((0, 0, 0, 0))
                continue
            t = inside / n  # white coverage
            a = round(255 * cover / n)
            colour = tuple(round(bg[i] * (1 - t) + WHITE[i] * t) for i in range(3))
            row += bytes((*colour, a))
        rows += row
    return bytes(rows)


def write_png(path: Path, w: int, h: int, raw: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Ibà">
  <rect width="64" height="64" rx="14" fill="#115e59"/>
  <circle cx="32" cy="11.9" r="4.9" fill="#fff"/>
  <rect x="27.1" y="20.8" width="9.8" height="21.6" rx="4.9" fill="#fff"/>
</svg>
"""

# Portrait launch images for current iPhones: (css width, css height, dpr).
IPHONES = [
    (430, 932, 3),
    (393, 852, 3),
    (428, 926, 3),
    (390, 844, 3),
    (375, 812, 3),
    (414, 896, 2),
    (375, 667, 2),
]


def main() -> int:
    (PUBLIC / "icon.svg").write_text(SVG, encoding="utf-8")
    icons = [
        ("icon-192.png", 192, 192, 0.66, True),
        ("icon-512.png", 512, 512, 0.66, True),
        # Maskable: full bleed, mark inside the 80% safe zone.
        ("icon-512-maskable.png", 512, 512, 0.5, False),
        ("apple-touch-icon.png", 180, 180, 0.66, False),  # iOS applies its own mask
    ]
    for name, w, h, scale, rounded in icons:
        write_png(PUBLIC / name, w, h, render(w, h, mark_scale=scale, rounded=rounded))
        print(f"  {name:<24} {w}x{h}")
    for cw, ch, dpr in IPHONES:
        w, h = cw * dpr, ch * dpr
        name = f"splash/launch-{w}x{h}.png"
        write_png(PUBLIC / name, w, h, render(w, h, mark_scale=0.28))
        print(f"  {name:<24} {w}x{h}  ({cw}x{ch} @{dpr}x)")
    print(f"Brand teal #{TEAL[0]:02x}{TEAL[1]:02x}{TEAL[2]:02x} on white mark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
