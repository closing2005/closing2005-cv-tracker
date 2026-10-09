"""One-command demo: generate a synthetic video, track it, save the result.

No camera, no downloads, no weights. Run it and watch the tracker work:

    python examples/synthetic_demo.py --out demo_out

Outputs:
    demo_out/input.mp4   - the synthetic video (two moving blobs, one occlusion)
    demo_out/tracked.mp4 - same video with boxes, IDs, trajectories and counts
    demo_out/summary.yaml - session summary
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import draw  # noqa: E402
from src.analytics import TrajectoryRecorder, print_summary, summarize  # noqa: E402
from src.detector import MotionDetector  # noqa: E402
from src.events import EventBus, SpeedingDetector  # noqa: E402
from src.tracker import MultiTracker  # noqa: E402
from src.zones import Tripwire  # noqa: E402

W, H, FPS, N_FRAMES = 640, 480, 30, 180


def make_video(path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N_FRAMES):
        img = np.zeros((H, W, 3), np.uint8)
        # Target A: moves right the whole time
        cv2.rectangle(img, (50 + 2 * f, 200), (100 + 2 * f, 250), (255, 255, 255), -1)
        # Target B: moves down, fully occluded for frames 70-84.
        # The tracker should keep its ID through the gap (stage-2 rescue).
        if f < 70 or f > 84:
            cv2.rectangle(img, (400, 40 + 2 * f), (450, 90 + 2 * f), (255, 255, 255), -1)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    in_path = os.path.join(args.out, "input.mp4")
    out_path = os.path.join(args.out, "tracked.mp4")
    print("generating synthetic video...")
    make_video(in_path)

    detector = MotionDetector(min_area=400)
    tracker = MultiTracker()
    wire = Tripwire("mid", (320, 40), (320, 440))
    bus = EventBus([
        SpeedingDetector(max_px_per_sec=400, fps=FPS),
    ])
    recorder = TrajectoryRecorder(os.path.join(args.out, "trajectories.csv"))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    cap = cv2.VideoCapture(in_path)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tracks = tracker.update(detector.detect(frame))
            fresh = []
            for e, i in wire.update(tracks):
                fresh.append({"type": e, "track_id": i, "wire": wire.name})
            fresh += bus.update(tracks, frame_idx)
            recorder.record(frame_idx, tracks)
            vw.write(draw(frame, tracks, [], [wire], fresh,
                          {"targets": len(tracks)}))
            frame_idx += 1
    finally:
        recorder.close()
        cap.release()
        vw.release()

    summary = summarize(list(tracker.tracks), list(bus.log), fps=FPS)
    print_summary(summary)
    print(f"tripwire '{wire.name}': forward={wire.forward} backward={wire.backward}")
    print(f"\ndone: {in_path}\n      {out_path}")


if __name__ == "__main__":
    main()
