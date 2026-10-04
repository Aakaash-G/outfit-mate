"""Download real clothing images from Unsplash for garment-tagging evaluation.

Creates data/eval/<category>/<n>.jpg with category names matching
core/vision.py CATEGORIES.

Run:  python scripts/setup_eval_data.py
"""
import os
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

# Bypass SSL issues on some Windows setups
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# (category_folder, filename, unsplash_url)
# Category names MUST match keys in core/vision.py CATEGORIES
EVAL_IMAGES = [
    # ── t-shirt ──
    ("t-shirt", "1.jpg",
     "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&q=80&fit=crop"),
    ("t-shirt", "2.jpg",
     "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=400&q=80&fit=crop"),
    ("t-shirt", "3.jpg",
     "https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?w=400&q=80&fit=crop"),

    # ── shirt ──
    ("shirt", "1.jpg",
     "https://images.unsplash.com/photo-1598032895397-b9472444bf93?w=400&q=80&fit=crop"),
    ("shirt", "2.jpg",
     "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&q=80&fit=crop"),
    ("shirt", "3.jpg",
     "https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?w=400&q=80&fit=crop"),

    # ── hoodie ──
    ("hoodie", "1.jpg",
     "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400&q=80&fit=crop"),
    ("hoodie", "2.jpg",
     "https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?w=400&q=80&fit=crop"),
    ("hoodie", "3.jpg",
     "https://images.unsplash.com/photo-1578768079052-aa76e52ff62e?w=400&q=80&fit=crop"),

    # ── jacket ──
    ("jacket", "1.jpg",
     "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&q=80&fit=crop"),
    ("jacket", "2.jpg",
     "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400&q=80&fit=crop"),
    ("jacket", "3.jpg",
     "https://images.unsplash.com/photo-1548883354-94bcfe321cbb?w=400&q=80&fit=crop"),

    # ── blazer ──
    ("blazer", "1.jpg",
     "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400&q=80&fit=crop"),
    ("blazer", "2.jpg",
     "https://images.unsplash.com/photo-1593030761757-71fae45fa0e7?w=400&q=80&fit=crop"),

    # ── jeans ──
    ("jeans", "1.jpg",
     "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&q=80&fit=crop"),
    ("jeans", "2.jpg",
     "https://images.unsplash.com/photo-1582552938357-32b906df40cb?w=400&q=80&fit=crop"),
    ("jeans", "3.jpg",
     "https://images.unsplash.com/photo-1565084888279-aca607ecce0c?w=400&q=80&fit=crop"),

    # ── trousers ──
    ("trousers", "1.jpg",
     "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400&q=80&fit=crop"),
    ("trousers", "2.jpg",
     "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400&q=80&fit=crop"),
    ("trousers", "3.jpg",
     "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400&q=80&fit=crop"),

    # ── shorts ──
    ("shorts", "1.jpg",
     "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=400&q=80&fit=crop"),
    ("shorts", "2.jpg",
     "https://images.unsplash.com/photo-1600185365926-3a2ce3cdb9eb?w=400&q=80&fit=crop"),
    ("shorts", "3.jpg",
     "https://images.unsplash.com/photo-1617952739858-28043cecdae3?w=400&q=80&fit=crop"),

    # ── sneakers ──
    ("sneakers", "1.jpg",
     "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&q=80&fit=crop"),
    ("sneakers", "2.jpg",
     "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=400&q=80&fit=crop"),
    ("sneakers", "3.jpg",
     "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400&q=80&fit=crop"),

    # ── boots ──
    ("boots", "1.jpg",
     "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?w=400&q=80&fit=crop"),
    ("boots", "2.jpg",
     "https://images.unsplash.com/photo-1605733160314-4fc7dac4bb16?w=400&q=80&fit=crop"),

    # ── dress ──
    ("dress", "1.jpg",
     "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&q=80&fit=crop"),
    ("dress", "2.jpg",
     "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=400&q=80&fit=crop"),
    ("dress", "3.jpg",
     "https://images.unsplash.com/photo-1612336307429-8a898d10e223?w=400&q=80&fit=crop"),

    # ── sweater ──
    ("sweater", "1.jpg",
     "https://images.unsplash.com/photo-1576871337632-b9aef4c17ab9?w=400&q=80&fit=crop"),
    ("sweater", "2.jpg",
     "https://images.unsplash.com/photo-1620799140188-3b2a02fd9a77?w=400&q=80&fit=crop"),

    # ── skirt ──
    ("skirt", "1.jpg",
     "https://images.unsplash.com/photo-1583496661160-fb5886a0aaaa?w=400&q=80&fit=crop"),
    ("skirt", "2.jpg",
     "https://images.unsplash.com/photo-1592301933927-35b597393c0a?w=400&q=80&fit=crop"),

    # ── sandals ──
    ("sandals", "1.jpg",
     "https://images.unsplash.com/photo-1603487742131-4160ec999306?w=400&q=80&fit=crop"),
    ("sandals", "2.jpg",
     "https://images.unsplash.com/photo-1562273138-f46be4ebdf33?w=400&q=80&fit=crop"),

    # ── formal shoes ──
    ("formal shoes", "1.jpg",
     "https://images.unsplash.com/photo-1614252369475-531eba835eb1?w=400&q=80&fit=crop"),
    ("formal shoes", "2.jpg",
     "https://images.unsplash.com/photo-1533867617858-e7b97e060509?w=400&q=80&fit=crop"),
]


def download_all():
    ok = fail = 0
    for cat, fname, url in EVAL_IMAGES:
        dest_dir = EVAL_DIR / cat
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname

        if dest.exists():
            print(f"  SKIP {cat}/{fname} (exists)")
            ok += 1
            continue

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            size_kb = len(data) / 1024
            print(f"  OK   {cat}/{fname}  ({size_kb:.0f} KB)")
            ok += 1
        except Exception as e:
            print(f"  FAIL {cat}/{fname}  -> {e}")
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed")
    print(f"Eval images saved to: {EVAL_DIR}")
    return fail


if __name__ == "__main__":
    print(f"Downloading {len(EVAL_IMAGES)} evaluation images ...\n")
    download_all()
