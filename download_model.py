"""
download_model.py
-----------------
Downloads MobileNetV2 model files for the wildlife classifier.
Run this ONCE on the Raspberry Pi before starting main.py.

Usage:
    python3 download_model.py
"""

import urllib.request
import urllib.error
from pathlib import Path

MODEL_DIR = Path("model")


FILES = {
    "mobilenet_v2_deploy.prototxt": (
        "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt"
    ),
    "mobilenet_v2.caffemodel": (
        "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel"
    ),
}



def download(url: str, dest: Path):
    print(f"Downloading {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"done ({size_mb:.1f} MB)")
    except urllib.error.URLError as e:
        print("FAILED")
        raise RuntimeError(f"Could not download {dest.name}: {e}") from e
    except Exception as e:
        print("FAILED")
        raise RuntimeError(f"Unexpected error downloading {dest.name}: {e}") from e


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    for filename, url in FILES.items():
        dest = MODEL_DIR / filename

        if dest.exists() and dest.stat().st_size > 0:
            print(f"{filename} already present - skipping")
            continue

        if dest.exists():
            dest.unlink()

        download(url, dest)

    print("\nAll model files are ready.")
    print("You can now run: python3 main.py")


if __name__ == "__main__":
    main()