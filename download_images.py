"""
Download real clothing images from Unsplash (free, no API key needed for direct URLs).
Saves to img/<category>/<filename>.jpg
"""
import urllib.request
import os
import ssl

# Bypass SSL verification issues on some Windows setups
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE = os.path.join(os.path.dirname(__file__), "img")

# Real Unsplash image URLs (free to use, resized to 400px width)
# Format: (category_folder, filename, unsplash_url)
IMAGES = [
    # ── Formal Shirts ──
    ("formal-shirts", "white-oxford.jpg",
     "https://images.unsplash.com/photo-1598032895397-b9472444bf93?w=400&q=80&fit=crop"),
    ("formal-shirts", "light-blue-poplin.jpg",
     "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&q=80&fit=crop"),
    ("formal-shirts", "charcoal-grey-slim.jpg",
     "https://images.unsplash.com/photo-1620012253295-c15cc3e65df4?w=400&q=80&fit=crop"),
    ("formal-shirts", "pale-pink-herringbone.jpg",
     "https://images.unsplash.com/photo-1607345366928-199ea26cfe3e?w=400&q=80&fit=crop"),
    ("formal-shirts", "navy-blue-french-cuff.jpg",
     "https://images.unsplash.com/photo-1588359348347-9bc6cbbb689e?w=400&q=80&fit=crop"),

    # ── Formal Pants ──
    ("formal-pants", "black-flat-front.jpg",
     "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400&q=80&fit=crop"),
    ("formal-pants", "navy-wool-slacks.jpg",
     "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400&q=80&fit=crop"),
    ("formal-pants", "charcoal-pleated.jpg",
     "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400&q=80&fit=crop"),
    ("formal-pants", "khaki-dress-chinos.jpg",
     "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=400&q=80&fit=crop"),
    ("formal-pants", "light-grey-check.jpg",
     "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&q=80&fit=crop"),

    # ── Casual Shirts ──
    ("casual-shirts", "red-black-flannel.jpg",
     "https://images.unsplash.com/photo-1608234808654-2a8875faa7fd?w=400&q=80&fit=crop"),
    ("casual-shirts", "olive-green-linen.jpg",
     "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400&q=80&fit=crop"),
    ("casual-shirts", "denim-button-down.jpg",
     "https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?w=400&q=80&fit=crop"),
    ("casual-shirts", "floral-short-sleeve.jpg",
     "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&q=80&fit=crop&flip=h"),
    ("casual-shirts", "mustard-corduroy.jpg",
     "https://images.unsplash.com/photo-1621072156002-e2fccdc0b176?w=400&q=80&fit=crop"),

    # ── Casual Jeans ──
    ("casual-jeans", "dark-wash-straight.jpg",
     "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&q=80&fit=crop"),
    ("casual-jeans", "light-blue-faded-slim.jpg",
     "https://images.unsplash.com/photo-1582552938357-32b906df40cb?w=400&q=80&fit=crop"),
    ("casual-jeans", "black-skinny-stretch.jpg",
     "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=400&q=80&fit=crop&sat=-100"),
    ("casual-jeans", "grey-distressed.jpg",
     "https://images.unsplash.com/photo-1604176354204-9268737828e4?w=400&q=80&fit=crop"),
    ("casual-jeans", "indigo-raw.jpg",
     "https://images.unsplash.com/photo-1565084888279-aca607ecce0c?w=400&q=80&fit=crop"),

    # ── Shorts ──
    ("shorts", "beige-chino.jpg",
     "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=400&q=80&fit=crop"),
    ("shorts", "navy-board.jpg",
     "https://images.unsplash.com/photo-1600185365926-3a2ce3cdb9eb?w=400&q=80&fit=crop"),
    ("shorts", "olive-cargo.jpg",
     "https://images.unsplash.com/photo-1617952739858-28043cecdae3?w=400&q=80&fit=crop"),
    ("shorts", "light-denim.jpg",
     "https://images.unsplash.com/photo-1598554747436-c9293d6a588f?w=400&q=80&fit=crop"),
    ("shorts", "black-sweat.jpg",
     "https://images.unsplash.com/photo-1562157873-818bc0726f68?w=400&q=80&fit=crop"),

    # ── Gym / Activewear ──
    ("gym", "black-graphic-tank.jpg",
     "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&q=80&fit=crop"),
    ("gym", "neon-yellow-tank.jpg",
     "https://images.unsplash.com/photo-1618354691438-25bc04584c23?w=400&q=80&fit=crop"),
    ("gym", "heather-grey-stringer.jpg",
     "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=400&q=80&fit=crop"),
    ("gym", "black-tapered-track.jpg",
     "https://images.unsplash.com/photo-1556906781-9a412961c28c?w=400&q=80&fit=crop"),
    ("gym", "grey-jogger.jpg",
     "https://images.unsplash.com/photo-1580906853149-f23000e4acfc?w=400&q=80&fit=crop"),
]


def download():
    ok = 0
    fail = 0
    for cat, fname, url in IMAGES:
        dest_dir = os.path.join(BASE, cat)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, fname)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            size_kb = len(data) / 1024
            print(f"  OK  {cat}/{fname}  ({size_kb:.0f} KB)")
            ok += 1
        except Exception as e:
            print(f"  FAIL  {cat}/{fname}  -> {e}")
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed")
    print(f"Images saved to: {BASE}")


if __name__ == "__main__":
    print(f"Downloading {len(IMAGES)} clothing images to img/ ...\n")
    download()
