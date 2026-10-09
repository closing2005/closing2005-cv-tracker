"""cv-tracker: real-time multi-object tracking from the command line.

    python -m src.main --source 0 --show
    python -m src.main --source traffic.mp4 --out runs/run1 --no-show

Pipeline: Detector -> MultiTracker -> Zones -> Events -> Recorder.
Per-stage timings are printed every 60 frames so bottlenecks are visible.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics import TrajectoryRecorder, print_summary, summarize
from src.detector import MotionDetector
from src.events import (CrowdingDetector, EventBus, LoiteringDetector,
                        SpeedingDetector, WrongWayDetector)
from src.tracker import MultiTracker
from src.zones import Tripwire, Zone

COLORS = [(56, 189, 248), (52, 211, 153), (251, 191, 36), (244, 114, 182),
          (167, 139, 250), (251, 113, 133)]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def draw(frame, tracks, zones, wires, events, timings):
    for z in zones:
        cv2.polylines(frame, [z.polygon], True, (148, 163, 184), 2)
        cv2.putText(frame, f"{z.name} in:{z.entries} out:{z.exits}",
                    tuple(z.polygon[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (148, 163, 184), 2)
    for w_ in wires:
        p1 = tuple(w_.p1.astype(int))
        p2 = tuple(w_.p2.astype(int))
        cv2.arrowedLine(frame, p1, p2, (250, 204, 21), 2)
        cv2.putText(frame, f"{w_.name} fwd:{w_.forward} bwd:{w_.backward}",
                    p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 204, 21), 2)
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        color = COLORS[t.id % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"#{t.id}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        pts = [(int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
               for b in t.history[-40:]]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(frame, a, b, color, 2)
    y = 24
    for name, ms in timings.items():
        cv2.putText(frame, f"{name}: {ms:.1f}ms", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 2)
        y += 22
    for e in events[-3:]:
        cv2.putText(frame, f"! {e['type']} #{e.get('track_id', '-')}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 113, 113), 2)
        y += 24
    return frame


def main():
    ap = argparse.ArgumentParser(description="Real-time multi-object tracker")
    ap.add_argument("--source", default="0",
                    help="camera index or video file path")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="runs/latest",
                    help="output dir for trajectories.csv and summary")
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = unlimited)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    det_c, tr_c = cfg["detector"], cfg["tracker"]

    try:
        src = int(args.source)
    except ValueError:
        src = args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"cannot open source: {args.source}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = MotionDetector(min_area=det_c.get("min_area", 500),
                              history=det_c.get("history", 500),
                              var_threshold=det_c.get("var_threshold", 16))
    tracker = MultiTracker(
        high_thresh=tr_c.get("high_thresh", 0.5),
        low_thresh=tr_c.get("low_thresh", 0.15),
        max_iou_dist=tr_c.get("max_iou_dist", 0.7),
        n_init=tr_c.get("n_init", 3),
        max_age=tr_c.get("max_age", 30),
        max_lost=tr_c.get("max_lost", 60))

    zones = [Zone(z["name"], z["polygon"]) for z in cfg.get("zones", [])]
    wires = [Tripwire(w["name"], w["p1"], w["p2"])
             for w in cfg.get("tripwires", [])]

    ev_c = cfg.get("events", {})
    detectors = []
    if ev_c.get("loitering", {}).get("enabled"):
        c = ev_c["loitering"]
        detectors.append(LoiteringDetector(c.get("radius_px", 40),
                                           c.get("min_frames", 150)))
    if ev_c.get("speeding", {}).get("enabled"):
        c = ev_c["speeding"]
        detectors.append(SpeedingDetector(c.get("max_px_per_sec", 300),
                                         fps=fps))
    if ev_c.get("wrong_way", {}).get("enabled"):
        c = ev_c["wrong_way"]
        detectors.append(WrongWayDetector(tuple(c.get("flow", [1, 0]))))
    if ev_c.get("crowding", {}).get("enabled") and zones:
        c = ev_c["crowding"]
        detectors.append(CrowdingDetector(zones[0], c.get("max_tracks", 5)))
    bus = EventBus(detectors)

    os.makedirs(args.out, exist_ok=True)
    recorder = TrajectoryRecorder(os.path.join(args.out, "trajectories.csv"))

    timings = {}
    frame_idx = 0
    done_tracks = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            dets = detector.detect(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(dets)
            t2 = time.perf_counter()
            fresh = []
            for z in zones:
                fresh += [dict(type=e, track_id=i, zone=z.name)
                          for e, i in z.update(tracks, frame_idx)]
            for w_ in wires:
                fresh += [dict(type=e, track_id=i, wire=w_.name)
                          for e, i in w_.update(tracks)]
            fresh += bus.update(tracks, frame_idx)
            t3 = time.perf_counter()
            recorder.record(frame_idx, tracks)
            timings = {"detect": (t1 - t0) * 1e3,
                       "track": (t2 - t1) * 1e3,
                       "zones+events": (t3 - t2) * 1e3}

            if args.show:
                draw(frame, tracks, zones, wires, fresh, timings)
                cv2.imshow("cv-tracker  (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"frame {frame_idx}: {len(tracks)} tracks, "
                      f"detect {timings['detect']:.1f}ms, "
                      f"track {timings['track']:.1f}ms")
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        done_tracks = list(tracker.tracks)
        recorder.close()
        cap.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # headless builds have no GUI backend

    summary = summarize(done_tracks, list(bus.log), fps=fps)
    print_summary(summary)
    with open(os.path.join(args.out, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f)
    print(f"saved trajectories + summary to {args.out}")


if __name__ == "__main__":
    main()
