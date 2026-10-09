"""Download the YOLO ONNX weights (not stored in git).

    python scripts/download_model.py [--model yolo11n] [--out weights/]

YOLO11n (~11MB) is the default: same output format as YOLOv8, runs in
cv2.dnn with no ultralytics dependency.
"""

import argparse
import os
import sys
import urllib.request

MODELS = {
    # name: (release tag, asset file, size MB approx)
    "yolo11n": ("v8.3.0", "yolo11n.onnx", 11),
    "yolov8n": ("v8.2.0", "yolov8n.onnx", 12),
}

BASE = "https://github.com/ultralytics/assets/releases/download"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n", choices=list(MODELS))
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()

    tag, asset, _ = MODELS[args.model]
    url = f"{BASE}/{tag}/{asset}"
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, asset)
    if os.path.exists(dest):
        print(f"already exists: {dest}")
        return
    print(f"downloading {url} ...")

    def hook(blocks, block_size, total):
        done = blocks * block_size
        pct = min(done / total * 100, 100) if total > 0 else 0
        sys.stdout.write(f"\r{done // 1024 // 1024}MB / "
                         f"{total // 1024 // 1024}MB ({pct:.0f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print(f"\nsaved to {dest}")


if __name__ == "__main__":
    main()
