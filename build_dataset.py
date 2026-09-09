"""
build_dataset.py — assemble a training set with the hard negatives your
current model is missing.

Your model scores a photo of a desk at 99.90% "accident". That happens when
the non-accident class in training had too little variety, so the network
learned "is this a real photograph?" instead of "is this a crash?".

This downloads Creative Commons images from Wikimedia Commons (no API key,
no account) into the layout train.py expects:

    dataset/
        accident/          <- your existing crash photos go here
        non_accident/      <- this script fills this

Usage
-----
    pip install requests pillow
    python build_dataset.py                 # ~1200 negatives
    python build_dataset.py --per-term 120  # more per category

Then copy your own accident photos into dataset/accident/ and run train.py.

Note: downloaded images are a starting point, not a finished dataset. The
single most valuable thing you can add is a few hundred photos taken on the
same phones your users will actually submit from — same camera, same lighting,
same motion blur.
"""

import argparse
import io
import os
import time

import requests
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia asks for a descriptive User-Agent identifying the tool.
HEADERS = {"User-Agent": "GoldenHour-dataset-builder/1.0 (student project)"}

# The categories the model currently gets wrong. Indoor scenes and ordinary
# traffic matter most: a model that has never seen a parked car will call one
# a collision.
NEGATIVE_TERMS = [
    # indoor — the desk photo failure mode
    "office desk computer", "classroom interior", "laptop on table",
    "living room interior", "shop interior", "library reading room",
    "kitchen interior", "restaurant interior", "hospital ward",
    # people — not casualties
    "person portrait outdoor", "group of people talking", "student studying",
    "man sitting chair", "woman walking street",
    # ordinary traffic — the dangerous near-misses
    "parked cars street", "traffic jam road", "city street traffic",
    "empty highway road", "car parking lot", "motorcycle parked",
    "bicycle parked street", "bus stop people", "road intersection",
    "truck on highway", "auto rickshaw india", "indian street traffic",
    # roadside things that are not crashes
    "roadworks construction", "pothole road", "road under repair",
    "broken down car roadside", "car maintenance garage",
    "traffic police officer", "road signs highway",
    # generic outdoor
    "buildings city view", "park trees path", "railway station platform",
]

POSITIVE_TERMS = [
    "traffic collision", "car accident damaged", "vehicle crash road",
    "road traffic accident", "motorcycle accident", "overturned vehicle",
    "car wreck", "bus accident", "truck accident road",
]


def search_images(term, limit, session):
    """Return direct image URLs from Wikimedia Commons for a search term."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": "filetype:bitmap %s" % term,
        "gsrlimit": min(limit, 50),
        "gsrnamespace": 6,                 # File: namespace
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": 640,                 # server-side thumbnail, small+fast
        "format": "json",
    }
    pages = None
    for attempt in range(4):
        try:
            r = session.get(API, params=params, headers=HEADERS, timeout=40)
            r.raise_for_status()
            pages = (r.json().get("query") or {}).get("pages") or {}
            break
        except Exception as exc:
            if attempt == 3:
                print("    search failed for %r: %s" % (term, str(exc)[:70]))
                return []
            time.sleep(2 ** attempt)          # 1s, 2s, 4s
    if pages is None:
        return []

    urls = []
    for page in pages.values():
        for info in page.get("imageinfo", []):
            if not str(info.get("mime", "")).startswith("image/"):
                continue
            url = info.get("thumburl") or info.get("url")
            if url:
                urls.append(url)
    return urls


def fetch(url, dest, session, min_side=200):
    """Download, verify and normalise one image to a 640px JPEG."""
    if os.path.exists(dest):
        return False
    content = None
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
            content = r.content
            break
        except Exception:
            if attempt == 2:
                return False
            time.sleep(1.5 * (attempt + 1))
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
        img = img.convert("RGB")
        if min(img.size) < min_side:            # too small to be useful
            return False
        img.thumbnail((640, 640), Image.LANCZOS)
        img.save(dest, "JPEG", quality=88)
        return True
    except Exception:
        return False


def collect(terms, out_dir, per_term, session, tag):
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for i, term in enumerate(terms, 1):
        urls = search_images(term, per_term, session)
        got = 0
        for j, url in enumerate(urls):
            if got >= per_term:
                break
            name = "%s_%02d_%03d.jpg" % (tag, i, j)
            if fetch(url, os.path.join(out_dir, name), session):
                got += 1
            time.sleep(0.08)                    # be polite to the API
        total += got
        print("  [%2d/%2d] %-32s %3d images" % (i, len(terms), term[:32], got))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset", help="output folder")
    ap.add_argument("--per-term", type=int, default=35,
                    help="images per search term")
    ap.add_argument("--positives", action="store_true",
                    help="also download accident images (use your own if you can)")
    args = ap.parse_args()

    session = requests.Session()

    neg_dir = os.path.join(args.out, "non_accident")
    print("=== downloading hard negatives -> %s ===" % neg_dir)
    n = collect(NEGATIVE_TERMS, neg_dir, args.per_term, session, "neg")
    print("  %d negatives\n" % n)

    if args.positives:
        pos_dir = os.path.join(args.out, "accident")
        print("=== downloading accident images -> %s ===" % pos_dir)
        p = collect(POSITIVE_TERMS, pos_dir, args.per_term, session, "pos")
        print("  %d positives\n" % p)

    pos_dir = os.path.join(args.out, "accident")
    os.makedirs(pos_dir, exist_ok=True)
    have_pos = len([f for f in os.listdir(pos_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    have_neg = len([f for f in os.listdir(neg_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    print("=" * 58)
    print("  accident      : %d images" % have_pos)
    print("  non_accident  : %d images" % have_neg)
    if have_pos == 0:
        print("\n  Put your crash photos in %s, then run train.py" % pos_dir)
    elif have_neg < have_pos * 0.8:
        print("\n  Negatives are thin next to positives. Run again with a"
              "\n  larger --per-term, or add your own photos.")
    else:
        print("\n  Balanced enough to train. Next: python train.py")


if __name__ == "__main__":
    main()
