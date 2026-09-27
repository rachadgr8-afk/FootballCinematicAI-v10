#!/usr/bin/env python3
"""
Rebuild the Android launcher icon to match the app's OWN brand identity
(emerald->teal gradient + white lightning bolt, exactly like the in-app header
logo). This only replaces the generic default Capacitor placeholder icon.
"""
import os
from PIL import Image, ImageDraw

# Brand colors taken from the app header: bg-gradient-to-br from-emerald-500 to-teal-600
EMERALD = (16, 185, 129)   # #10B981
TEAL    = (13, 148, 136)   # #0D9488
WHITE   = (255, 255, 255)

BASE = "/home/user/work/football_app/android/app/src/main/res"
DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}

# Bolt polygon in a 24x24 viewBox (classic zap shape)
BOLT_24 = [(13, 2), (4, 14), (11, 14), (11, 22), (20, 10), (13, 10)]

SS = 4  # supersampling factor


def diag_gradient(size, c1, c2):
    """Diagonal (top-left -> bottom-right) linear gradient like bg-gradient-to-br."""
    n = size
    grad = Image.new("RGB", (n, n))
    px = grad.load()
    for y in range(n):
        for x in range(n):
            t = (x + y) / (2 * (n - 1)) if n > 1 else 0
            px[x, y] = (
                int(c1[0] + (c2[0] - c1[0]) * t),
                int(c1[1] + (c2[1] - c1[1]) * t),
                int(c1[2] + (c2[2] - c1[2]) * t),
            )
    return grad


def bolt_mask(size, scale=1.0):
    """Anti-aliased white bolt mask on transparent, bolt occupying `scale` of canvas."""
    S = size * SS
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    # scale bolt from 24x24 space into the canvas with a margin
    span = 24 / scale
    off = (24 - span) / 2  # center offset in 24-space
    pts = [(((x - off) / span) * S, ((y - off) / span) * S) for (x, y) in BOLT_24]
    d.polygon(pts, fill=255)
    return m.resize((size, size), Image.LANCZOS)


def build_full_icon(size, round_icon=False):
    """Legacy square/round launcher: gradient background + white bolt."""
    S = size * SS
    base = diag_gradient(S, EMERALD, TEAL).convert("RGBA")
    if round_icon:
        mask = Image.new("L", (S, S), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, S - 1, S - 1), fill=255)
        base.putalpha(mask)
    bolt = bolt_mask(S, scale=0.62)
    white = Image.new("RGBA", (S, S), WHITE + (255,))
    base = Image.composite(white, base, bolt)
    if not round_icon:
        # rounded-square corners (legacy pre-Android-8 look)
        r = int(S * 0.22)
        corner = Image.new("L", (S, S), 0)
        ImageDraw.Draw(corner).rounded_rectangle((0, 0, S - 1, S - 1), radius=r, fill=255)
        base.putalpha(corner)
    return base.resize((size, size), Image.LANCZOS)


def build_foreground(size):
    """Adaptive-icon foreground: bolt centered inside the 66% safe zone."""
    S = size * SS
    fg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bolt = bolt_mask(S, scale=0.42)  # keeps bolt within adaptive safe zone
    white = Image.new("RGBA", (S, S), WHITE + (255,))
    fg = Image.composite(white, fg, bolt)
    return fg.resize((size, size), Image.LANCZOS)


for dens, factor in DENSITIES.items():
    d = os.path.join(BASE, f"mipmap-{dens}")
    os.makedirs(d, exist_ok=True)
    legacy = int(48 * factor)
    fg = int(108 * factor)
    build_full_icon(legacy, False).save(os.path.join(d, "ic_launcher.png"))
    build_full_icon(legacy, True).save(os.path.join(d, "ic_launcher_round.png"))
    build_foreground(fg).save(os.path.join(d, "ic_launcher_foreground.png"))
    print(f"[{dens}] legacy={legacy}px fg={fg}px")

print("Icon generation complete.")
