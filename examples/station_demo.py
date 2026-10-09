"""真实视频演示：人流跟踪 + 绊线计数。

YOLO person 检测 -> MultiTracker(ReID) -> 横向绊线计数 -> 标注视频。

    python examples/station_demo.py --source crowd.mp4 --out runs/crowd_result.mp4
"""

import sys, os, time
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from src.yolo_detector import YoloDetector
from src.tracker import MultiTracker, TrackState
from src.zones import Tripwire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--out", default="runs/station_result.mp4")
    ap.add_argument("--weights", default="weights/yolo11n.onnx")
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"weights not found: {args.weights}")
        print("run: python scripts/download_model.py")
        sys.exit(1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    det = YoloDetector(args.weights, conf_thresh=args.conf, classes=["person"])
    tracker = MultiTracker(n_init=3, max_lost=30, use_reid=True, reid_thresh=0.5)
    cap = cv2.VideoCapture(args.source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 横向绊线：画面中间，统计上下穿越
    wire = Tripwire("main", (0, h // 2), (w, h // 2))

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t0 = time.perf_counter()
    max_tracks = 0
    for fi in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        boxes = det.detect(frame)
        tracks = tracker.update(boxes, frame=frame)
        confirmed = [t for t in tracks if t.state == TrackState.CONFIRMED]
        max_tracks = max(max_tracks, len(confirmed))
        wire.update(confirmed)
        for t in confirmed:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"#{t.id}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # 画绊线 + 计数面板
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 2)
        cv2.putText(frame, f"up: {wire.forward}  down: {wire.backward}  live: {len(confirmed)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        vw.write(frame)
        if fi % 100 == 0:
            print(f"frame {fi}/{n}", flush=True)
    vw.release()
    cap.release()
    dt = time.perf_counter() - t0
    print(f"done: {n} frames in {dt:.0f}s ({n/dt:.1f} fps)")
    print(f"tripwire forward={wire.forward} backward={wire.backward} max_live={max_tracks}")

if __name__ == "__main__":
    main()
