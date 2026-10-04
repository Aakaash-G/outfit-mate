"""
Generate sample clothing placeholder images for OutfitMate.
Saves JPEGs to static/images/<category>/<filename>.jpg
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

BASE = Path(__file__).parent / "static" / "images"

# (category_dir, filename, label, bg_color, text_color)
ITEMS = [
    # Formal Shirts
    ("formal-shirts", "white-oxford.jpg",         "Classic White\nOxford",          "#F5F5F5", "#222222"),
    ("formal-shirts", "light-blue-poplin.jpg",    "Light Blue\nPoplin",             "#B0C8E8", "#1A2E4A"),
    ("formal-shirts", "charcoal-grey-slim.jpg",   "Charcoal Grey\nSlim-Fit",        "#4A4A4A", "#F0F0F0"),
    ("formal-shirts", "pale-pink-herringbone.jpg","Pale Pink\nHerringbone",          "#F2C2CE", "#5A2535"),
    ("formal-shirts", "navy-blue-french-cuff.jpg","Navy Blue\nFrench Cuff",          "#1B2A4A", "#D0D8F0"),

    # Formal Pants
    ("formal-pants", "black-flat-front.jpg",      "Black\nFlat-Front",              "#1A1A1A", "#E0E0E0"),
    ("formal-pants", "navy-wool-slacks.jpg",       "Navy Wool-\nBlend Slacks",       "#1C2B4A", "#C8D4F0"),
    ("formal-pants", "charcoal-pleated.jpg",       "Charcoal Grey\nPleated",         "#5A5A5A", "#F0F0F0"),
    ("formal-pants", "khaki-dress-chinos.jpg",     "Khaki Dress\nChinos",            "#C8A96E", "#3A2800"),
    ("formal-pants", "light-grey-check.jpg",       "Light Grey\nCheckered",          "#C8C8C8", "#333333"),

    # Casual Shirts
    ("casual-shirts", "red-black-flannel.jpg",    "Red & Black\nFlannel",           "#8B1A1A", "#F5E6E6"),
    ("casual-shirts", "olive-green-linen.jpg",    "Olive Green\nLinen",             "#6B7A3A", "#F0EDD0"),
    ("casual-shirts", "denim-button-down.jpg",    "Denim\nButton-Down",             "#4A6FA5", "#E8EFFF"),
    ("casual-shirts", "floral-short-sleeve.jpg",  "Floral\nShort-Sleeve",           "#E87060", "#FFF5F0"),
    ("casual-shirts", "mustard-corduroy.jpg",     "Mustard Yellow\nCorduroy",       "#D4A017", "#3A2800"),

    # Casual Jeans
    ("casual-jeans", "dark-wash-straight.jpg",    "Dark Wash\nStraight Leg",        "#2C3E6E", "#D0D8F0"),
    ("casual-jeans", "light-blue-faded-slim.jpg", "Light Blue\nFaded Slim",         "#8AAFD4", "#1A2E4A"),
    ("casual-jeans", "black-skinny-stretch.jpg",  "Black Skinny\nStretch",          "#1A1A1A", "#E0E0E0"),
    ("casual-jeans", "grey-distressed.jpg",       "Grey Distressed\nDenim",         "#7A7A7A", "#F0F0F0"),
    ("casual-jeans", "indigo-raw.jpg",            "Indigo Raw\nDenim",              "#2E3A6A", "#C8D0F0"),

    # Shorts
    ("shorts", "beige-chino.jpg",                 "Beige Chino\nShorts",            "#D4C5A0", "#3A2800"),
    ("shorts", "navy-board.jpg",                  "Navy Blue\nBoard Shorts",        "#1C2B5A", "#C8D4F0"),
    ("shorts", "olive-cargo.jpg",                 "Olive Cargo\nShorts",            "#5A6B2A", "#F0EDD0"),
    ("shorts", "light-denim.jpg",                 "Light Wash\nDenim Shorts",       "#A8C0DA", "#1A2E4A"),
    ("shorts", "black-sweat.jpg",                 "Black Sweat\nShorts",            "#1A1A1A", "#E0E0E0"),

    # Gym / Activewear
    ("gym", "black-graphic-tank.jpg",             "Black Graphic\nSleeveless",      "#111111", "#E0E0E0"),
    ("gym", "neon-yellow-tank.jpg",               "Neon Yellow\nMoisture-Wicking",  "#D4F000", "#1A1A00"),
    ("gym", "heather-grey-stringer.jpg",          "Heather Grey\nStringer",         "#9A9A9A", "#1A1A1A"),
    ("gym", "black-tapered-track.jpg",            "Black Tapered\nTrack Pants",     "#1A1A1A", "#E0E0E0"),
    ("gym", "grey-jogger.jpg",                    "Grey Jogger\nSweatpants",        "#7A7A7A", "#F0F0F0"),
]

# Category display labels for the tag
CAT_LABELS = {
    "formal-shirts": "Formal Shirt",
    "formal-pants":  "Formal Pants",
    "casual-shirts": "Casual Shirt",
    "casual-jeans":  "Casual Jeans",
    "shorts":        "Shorts",
    "gym":           "Activewear",
}

W, H = 400, 500

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def lighten(rgb, factor=0.15):
    return tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)

def darken(rgb, factor=0.2):
    return tuple(max(0, int(c * (1 - factor))) for c in rgb)

def make_image(category, filename, label, bg_hex, text_hex):
    dest = BASE / category / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    bg = hex_to_rgb(bg_hex)
    fg = hex_to_rgb(text_hex)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Gradient-like effect: slightly lighter at top, darker at bottom
    for y in range(H):
        ratio = y / H
        r = int(bg[0] * (1 - ratio * 0.18))
        g = int(bg[1] * (1 - ratio * 0.18))
        b = int(bg[2] * (1 - ratio * 0.18))
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Subtle clothing silhouette rectangle
    sil_color = darken(bg, 0.12) if sum(bg) > 300 else lighten(bg, 0.12)
    draw.rounded_rectangle([60, 60, 340, 380], radius=30, fill=sil_color)

    # Inner detail lines (texture suggestion)
    detail_color = darken(bg, 0.08) if sum(bg) > 300 else lighten(bg, 0.08)
    for y_off in range(100, 360, 20):
        draw.line([(80, y_off), (320, y_off)], fill=detail_color, width=1)

    # Category badge
    cat_label = CAT_LABELS.get(category, category)
    badge_bg = (*fg, 200)
    draw.rounded_rectangle([20, 20, 20 + len(cat_label) * 9 + 16, 44], radius=8, fill=fg)
    try:
        font_small = ImageFont.truetype("arial.ttf", 14)
        font_main  = ImageFont.truetype("arialbd.ttf", 22)
    except Exception:
        font_small = ImageFont.load_default()
        font_main  = ImageFont.load_default()

    draw.text((28, 24), cat_label, fill=bg, font=font_small)

    # Main label text
    lines = label.split("\n")
    y_start = 400
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font_main)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        draw.text((x, y_start + i * 30), line, fill=fg, font=font_main)

    img.save(str(dest), "JPEG", quality=92)
    print(f"  ✓ {category}/{filename}")

if __name__ == "__main__":
    print(f"Generating {len(ITEMS)} clothing images...")
    for item in ITEMS:
        make_image(*item)
    print(f"\nDone! Images saved to: {BASE}")
