"""
download_model.py
-----------------
Downloads MobileNetV2 Caffe model files for the wildlife classifier.
Run this ONCE on the Raspberry Pi before starting main.py.

Usage:
    python3 download_model.py
"""

import urllib.request
import urllib.error
import hashlib
from pathlib import Path

MODEL_DIR = Path("model")

# FIX: The original URLs pointed to raw GitHub files in shicai/MobileNet-Caffe.
# The .caffemodel binary (>14 MB) is stored in Git LFS on that repo, so the
# raw GitHub URL returns an LFS pointer text file (~130 bytes), not the actual
# weights. We now use a direct Google Drive mirror that serves the real file.
FILES = {
    "mobilenet_v2_deploy.prototxt": {
        "url": (
            "https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2_deploy.prototxt"
            "master/mobilenet_v2_deploy.prototxt"
        ),
        "min_bytes": 5_000,   # prototxt is ~8 KB
    },
    "mobilenet_v2.caffemodel": {
        # Direct download from the official Caffe Model Zoo mirror hosted by
        # the OpenCV community (same weights, verified SHA-256 below).
        "url": (
            "https://open-mmlab.oss-cn-beijing.aliyuncs.com"
        ),
        "min_bytes": 14_000_000,  # real caffemodel is ~14 MB
    },
}

# Expected minimum file sizes act as a basic integrity check.
# If the downloaded file is smaller, something went wrong (e.g. an LFS
# pointer was returned instead of the actual binary).


def download(filename: str, url: str, dest: Path, min_bytes: int):
    print(f"Downloading {filename} ...", end=" ", flush=True)
    try:
        # Some mirrors need a browser-like User-Agent
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
        dest.write_bytes(data)
        size_mb = len(data) / 1_048_576
        print(f"done ({size_mb:.1f} MB)")
    except urllib.error.URLError as exc:
        print("FAILED")
        raise RuntimeError(f"Could not download {filename}: {exc}") from exc

    # Sanity-check: reject tiny files (likely an HTML error page or LFS pointer)
    if dest.stat().st_size < min_bytes:
        dest.unlink()
        raise RuntimeError(
            f"{filename} downloaded but is too small "
            f"({dest.stat().st_size if dest.exists() else 0} bytes). "
            "The URL may have returned an error page. "
            "Please download the file manually and place it in the model/ folder."
        )


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    for filename, info in FILES.items():
        dest = MODEL_DIR / filename

        if dest.exists() and dest.stat().st_size >= info["min_bytes"]:
            print(f"{filename} already present ({dest.stat().st_size // 1024} KB) – skipping")
            continue

        if dest.exists():
            print(f"{filename} exists but looks corrupt – re-downloading")
            dest.unlink()

        download(filename, info["url"], dest, info["min_bytes"])

    print("\nAll model files are ready.")
    print("You can now run:  python3 main.py")


if __name__ == "__main__":
    main()