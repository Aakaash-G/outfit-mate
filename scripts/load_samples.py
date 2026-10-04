"""Load all photos from a folder straight into the wardrobe database (no upload button needed).

Run from the outfit_mate folder:
    python scripts/load_samples.py                 # loads sample_wardrobe/
    python scripts/load_samples.py my_photos       # loads any other folder
    python scripts/load_samples.py --fresh         # empties the wardrobe first, then loads
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from core import db  # noqa: E402
from core.vision import describe, get_clip, save_image, tag_image  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = Path(args[0]) if args else ROOT / "img/sample_wardrobe"
    if not folder.is_absolute():
        folder = (Path.cwd() / folder) if (Path.cwd() / folder).exists() else ROOT / folder
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS) if folder.exists() else []
    if not photos:
        print(f"No photos found in {folder}")
        return

    db.init()
    existing = db.list_garments()
    if "--fresh" in sys.argv:
        for g in existing:
            db.delete_garment(g["id"])
        print(f"Removed {len(existing)} old garments.")
    elif existing:
        print(f"Note: the wardrobe already has {len(existing)} garments. New ones will be added on top.")
        print("      Use --fresh to start from an empty wardrobe.\n")

    print("Loading CLIP (the first run downloads it, please wait)...")
    clip = get_clip()
    for i, p in enumerate(photos, 1):
        try:
            tags = tag_image(Image.open(p), clip)
            path = save_image(tags.pop("clean_image"))
            gid = db.add_garment(tags, path)
            print(f"[{i:>2}/{len(photos)}] {p.name:<24} -> #{gid} {describe(tags)} (formality {tags['formality']})")
        except Exception as ex:
            print(f"[{i:>2}/{len(photos)}] {p.name:<24} FAILED: {type(ex).__name__}: {ex}")
    print(f"\nDone. The wardrobe now has {len(db.list_garments())} garments. Run: streamlit run app.py")


if __name__ == "__main__":
    main()