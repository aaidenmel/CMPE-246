"""
download_model.py
-----------------
Downloads MobileNetV2 (ImageNet) model files for the wildlife classifier.
Run this ONCE on the Raspberry Pi before starting main.py.

Usage:
    python download_model.py
"""

import urllib.request
import hashlib
from pathlib import Path

MODEL_DIR = Path("model")

FILES = {
    "mobilenet_v2_deploy.prototxt": (
        "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/"
        "mobilenet_v2_deploy.prototxt",
        None,   # no hash check for prototxt
    ),
    "mobilenet_v2.caffemodel": (
        "https://github.com/shicai/MobileNet-Caffe/releases/download/"
        "1.0/mobilenet_v2.caffemodel",
        None,
    ),
    # ImageNet class labels (synset)
    "synset_words.txt": (
        "https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/"
        "data/ilsvrc12/synset_words.txt",
        None,
    ),
}


def download(url: str, dest: Path):
    print(f"  Downloading {dest.name} …", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1_048_576
    print(f"done  ({size_mb:.1f} MB)")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    for filename, (url, _) in FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"  {filename} already present – skipping")
            continue
        download(url, dest)
    print("\nAll model files ready.  You can now run:  python main.py")


if __name__ == "__main__":
    main()
