"""Measure garment-tagging accuracy and speed.

By default it tests the 30 photos in sample_wardrobe/ using sample_wardrobe/labels.csv.
To test your own photos instead, put them in folders named after the category, e.g.
    data/eval/shirt/1.jpg   data/eval/jeans/a.png
(folder names must match categories in core/vision.py CATEGORIES)

Run:  python scripts/evaluate.py
"""
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from core.config import DATA_DIR  # noqa: E402
from core.vision import CATEGORIES, get_clip, tag_image  # noqa: E402

EVAL_DIR = DATA_DIR / "eval"
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_wardrobe"


def labelled_images():
    """[(image path, true category)] from data/eval/ if it exists, else from sample_wardrobe/labels.csv."""
    items = []
    if EVAL_DIR.exists():
        print(f"Using your photos in {EVAL_DIR}")
        for folder in sorted(p for p in EVAL_DIR.iterdir() if p.is_dir()):
            true = folder.name.replace("_", " ")
            if true not in CATEGORIES:
                print(f"skipping '{folder.name}' (not a known category)")
                continue
            items += [(f, true) for f in sorted(folder.iterdir())
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    elif (SAMPLE_DIR / "labels.csv").exists():
        print(f"Using the sample wardrobe in {SAMPLE_DIR}")
        with open(SAMPLE_DIR / "labels.csv") as fh:
            items = [(SAMPLE_DIR / r["filename"], r["category"]) for r in csv.DictReader(fh)]
    return items


def main():
    items = labelled_images()
    if not items:
        print("No labelled images found. Add data/eval/<category>/ folders or keep sample_wardrobe/.")
        return
    clip = get_clip()
    total = correct = slot_correct = 0
    per_class = defaultdict(lambda: [0, 0])
    confusions = Counter()
    times = []
    for f, true in items:
        t = time.perf_counter()
        pred = tag_image(Image.open(f), clip)
        times.append(time.perf_counter() - t)
        total += 1
        ok = pred["category"] == true
        correct += ok
        slot_correct += pred["slot"] == CATEGORIES[true][0]
        per_class[true][0] += ok
        per_class[true][1] += 1
        if not ok:
            confusions[(true, pred["category"])] += 1
    if not total:
        print("No images found.")
        return
    print(f"\nImages evaluated     : {total}")
    print(f"Category accuracy    : {correct / total:.1%}")
    print(f"Slot accuracy        : {slot_correct / total:.1%}   (top / bottom / shoes ...)")
    print(f"Mean tagging time    : {sum(times) / len(times) * 1000:.0f} ms per image (incl. background removal)")
    print("\nPer-class accuracy:")
    for c, (ok, n) in sorted(per_class.items()):
        print(f"  {c:<14} {ok}/{n}  ({ok / n:.0%})")
    if confusions:
        print("\nMost common mistakes (true -> predicted):")
        for (t, p), n in confusions.most_common(8):
            print(f"  {t} -> {p}: {n}")


if __name__ == "__main__":
    main()