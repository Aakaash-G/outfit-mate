"""Bulk-add the downloaded clothing images (from img/) to the wardrobe DB.

This is equivalent to uploading them through the Streamlit UI one-by-one.

Run:  python scripts/bulk_add_wardrobe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from core import db  # noqa: E402
from core.vision import get_clip, save_image, tag_image, describe  # noqa: E402

IMG_DIR = Path(__file__).resolve().parent.parent / "img"


def main():
    if not IMG_DIR.exists():
        print(f"Download images first:  python download_images.py")
        return

    db.init()

    # Check what's already in the wardrobe
    existing = {g["name"] for g in db.list_garments()}
    print(f"Already in wardrobe: {len(existing)} garments\n")

    image_files = sorted(
        f for d in IMG_DIR.iterdir() if d.is_dir()
        for f in d.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )

    if not image_files:
        print("No images found in img/")
        return

    print(f"Found {len(image_files)} images to process")
    print("Loading CLIP model...")
    clip = get_clip()

    added = skipped = failed = 0
    for i, f in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {f.relative_to(IMG_DIR)}")
        try:
            img = Image.open(f)
            tags = tag_image(img, clip)
            clean_img = tags.pop("clean_image")

            # Skip if something with same name exists
            if tags["name"] in existing:
                print(f"  SKIP (already exists: {tags['name']})")
                skipped += 1
                continue

            path = save_image(clean_img)
            db.add_garment(tags, path)
            print(f"  -> {describe(tags)} | formality {tags['formality']} | "
                  f"confidence {tags['confidence']:.2f}")
            added += 1
            existing.add(tags["name"])
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Added:   {added}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    print(f"Total in wardrobe: {len(db.list_garments())}")


if __name__ == "__main__":
    main()
