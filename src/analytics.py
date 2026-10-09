"""Analytics: trajectory recording, session stats and tracker self-check.

Two jobs:
  1. Record every confirmed track's trajectory to CSV for offline analysis.
  2. Summarize the run: counts, dwell, speeds -- plus a tracker
     self-diagnostic (avg track length, fragmentation) so you can tell
     whether the numbers are trustworthy or the tracker was hallucinating.
"""

from __future__ import annotations

import csv
import math
import os


class TrajectoryRecorder:
    def __init__(self, path):
        self.path = path
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2",
                          "cx", "cy", "vx", "vy"])
        self.rows = 0

    def record(self, frame_idx, tracks):
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            vx, vy = t.kf.velocity()
            self._w.writerow([frame_idx, t.id,
                              round(x1, 1), round(y1, 1),
                              round(x2, 1), round(y2, 1),
                              round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                              round(vx, 2), round(vy, 2)])
            self.rows += 1

    def close(self):
        self._f.close()


def summarize(tracks_done, events, fps=30):
    """Build a human-readable session summary dict.

    tracks_done: final Track objects (history intact after deletion).
    """
    total = len(tracks_done)
    lifetimes = [len(t.history) for t in tracks_done]
    avg_life = sum(lifetimes) / total if total else 0

    speeds = []
    for t in tracks_done:
        h = t.history
        if len(h) >= 2:
            d = sum(
                math.hypot((b[0] + b[2] - a[0] - a[2]) / 2,
                           (b[1] + b[3] - a[1] - a[3]) / 2)
                for a, b in zip(h[:-1], h[1:])
            )
            speeds.append(d / (len(h) - 1) * fps)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    # Fragmentation heuristic: many short tracks ~= one real target
    # shattered by occlusion. <8 frames is suspicious for a confirmed track.
    fragments = sum(1 for L in lifetimes if L < 8)
    quality = "good"
    if total and fragments / total > 0.5:
        quality = "poor (high fragmentation -- tune detector/tracker)"
    elif total and fragments / total > 0.25:
        quality = "fair (some fragmentation)"

    by_type = {}
    for e in events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1

    return {
        "tracks": total,
        "avg_track_lifetime_frames": round(avg_life, 1),
        "avg_speed_px_per_sec": round(avg_speed, 1),
        "fragmented_tracks": fragments,
        "track_quality": quality,
        "events": by_type,
    }


def print_summary(summary):
    print("---- session summary ----")
    print(f"tracks seen          : {summary['tracks']}")
    print(f"avg lifetime (frames): {summary['avg_track_lifetime_frames']}")
    print(f"avg speed (px/s)     : {summary['avg_speed_px_per_sec']}")
    print(f"fragmented tracks    : {summary['fragmented_tracks']}")
    print(f"track quality        : {summary['track_quality']}")
    print(f"events               : {summary['events'] or 'none'}")
