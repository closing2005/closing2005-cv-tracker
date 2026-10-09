# cv-tracker

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](#tests)
[![GitHub stars](https://img.shields.io/github/stars/closing2005/closing2005-cv-tracker.svg)](https://github.com/closing2005/closing2005-cv-tracker/stargazers)

English | [中文](README.zh-CN.md) | [术语表 Glossary](GLOSSARY.md)

</div>

Real-time multi-object tracking for static cameras. Detection → Kalman-filtered
tracking → zones & tripwires → behavioral events — in one CLI, with **zero
weight files** and **zero SciPy**.

```
frame → [Detector] → [MultiTracker] → [Zones / Tripwires] → [Events] → CSV + summary
              MOG2         ByteTrack-style        entry/exit    loitering
              motion       2-stage assoc.         counting      speeding
                           + Kalman                           wrong-way
                           + Hungarian                        crowding
```

## Why this isn't another toy tracker

- **Two-stage association (ByteTrack idea).** High-confidence detections match
  first; leftover tracks get a second chance against *low*-confidence
  detections. Briefly occluded targets are rescued instead of reborn with new
  IDs — the single biggest source of ID switches in naive trackers, handled.
- **Kalman filter with a constant-velocity model**, where the process noise
  scales with target size. Large, fast boxes get a wider gate than small,
  distant ones — no per-scene tuning.
- **Hungarian algorithm implemented from scratch** (`src/hungarian.py`):
  O(n³) Kuhn-Munkres, rectangular matrices, no SciPy, no black box.
- **A track lifecycle that kills flicker ghosts**:
  `Tentative → Confirmed → Lost → Deleted`. Only confirmed tracks are
  reported, counted, or fed to event detectors.
- **Behavioral events, not just boxes.** Loitering, speeding, wrong-way
  motion, and crowding — computed from track history alone, no extra model.
  Every event carries its evidence (track id, frame, measured value), so
  results are auditable, not magic.
- **Appearance re-ID (optional).** Zero-weight HSV-histogram embeddings let
  tracks survive *long* occlusions where Kalman prediction has drifted.
  Enable with `tracker.use_reid: true`.
- **Multi-camera handover.** `MultiCameraTracker` keeps one global identity
  across views via a shared appearance gallery — "the person in cam 2 is the
  same one who left cam 1."
- **Live web dashboard.** `--dashboard 8080`: real-time counts, FPS and event
  stream in the browser. Stdlib only, no new dependencies.
- **Tracker self-diagnostics.** The session summary reports a fragmentation
  rate and a quality verdict, so you know whether to trust the numbers or
  retune.
- **Pluggable detectors.** Ships with a weight-free MOG2 motion detector for
  static cameras; or switch to YOLO11n via `cv2.dnn` (see [docs/yolo_detector.md](docs/yolo_detector.md)
  for the theory, [docs/comparison.md](docs/comparison.md) for MOG2 vs YOLO
  benchmarks). Subclass `Detector` to plug in anything else without touching
  the tracker.

## Quickstart

```bash
pip install -r requirements.txt

# webcam, live visualization
python -m src.main --source 0 --show

# video file, headless, save trajectories + summary
python -m src.main --source traffic.mp4 --no-show --out runs/run1 --max-frames 900

# YOLO detector instead of motion (download weights first, ~11MB)
python scripts/download_model.py
# then set detector.type: "yolo" in config.yaml

# live web dashboard at http://127.0.0.1:8080
python -m src.main --source 0 --dashboard 8080 --no-show

# run the test suite (no camera needed)
python -m pytest tests/ -q
```

## One-command demo

No camera? No problem. This generates a synthetic video (two targets, one
15-frame occlusion), tracks it end to end, and saves an annotated result video:

```bash
python examples/synthetic_demo.py --out demo_out
# demo_out/input.mp4    - the synthetic input
# demo_out/tracked.mp4  - boxes, IDs, trajectories, tripwire counts drawn on
# demo_out/summary.yaml - session summary
```

## Real-world demo

`examples/station_demo.py` runs the full pipeline on real footage —
YOLO person detection → tracker with re-ID → tripwire counting:

```bash
python scripts/download_model.py   # weights/yolo11n.onnx, once
python examples/station_demo.py --source market.mp4 --out runs/market.mp4 \
    --conf 0.25 --max-lost 90 --reid-thresh 0.4
# --exclude "x1,y1,x2,y2" masks billboards / false-positive zones
```

Tested on an 18-second fish-market clip (fixed camera, ~12 people):
8 up / 12 down across the tripwire, stable IDs throughout.

![fish-market tracking demo](docs/market_demo%20(1).mp4)

## Benchmark

Measured on a 640×480 @ 30fps synthetic stream, CPU only (no GPU, no weights):

| Stage | Time per frame |
|---|---|
| Detection (MOG2 + morphology + contours) | ~5.5 ms |
| Tracking (Kalman predict + 2-stage Hungarian) | ~0.4 ms |
| Zones + events | <0.1 ms |

Comfortably real-time; detection dominates, so a DNN detector is where you'd
spend budget first.

Outputs in `--out/`:

| File | Contents |
|---|---|
| `trajectories.csv` | `frame, track_id, x1, y1, x2, y2, cx, cy, vx, vy` per confirmed track |
| `summary.yaml` | counts, speeds, events, and the track-quality verdict |

## Example session

```
frame 60: 2 tracks, detect 5.7ms, track 0.4ms
frame 120: 2 tracks, detect 5.1ms, track 0.3ms
---- session summary ----
tracks seen          : 4
avg lifetime (frames): 40.8
avg speed (px/s)     : 68.6
fragmented tracks    : 1
track quality        : good
events               : none
saved trajectories + summary to runs/run1
```

Per-stage timings print every 60 frames, so you can see exactly where the
milliseconds go.

## How it works

**Detection.** `MotionDetector` runs MOG2 background subtraction, drops
shadows, applies open/close morphology, and filters contours by area. Each
blob is scored by size and fill-ratio (solid objects outrank wispy noise),
largest first — which helps the matcher prioritize.

**Tracking.** Every frame, each live track is Kalman-predicted forward. Then:

1. *Stage 1* — match tracks against high-confidence detections by IoU cost,
   solved optimally with the Hungarian algorithm (gated at `max_iou_dist`).
2. *Stage 2* — match the leftovers against low-confidence detections. This
   is the ByteTrack insight: a weak glimpse of an occluded target is better
   than a new ID.
3. Unmatched high-confidence detections seed new tentative tracks; a track
   needs `n_init` consecutive hits to become confirmed.

Lost tracks are kept alive for `max_lost` frames on motion prediction alone,
so brief occlusions don't fragment identities.

**Zones & tripwires.** Polygon zones count entries/exits and measure dwell
time per track. Directed tripwires report crossing *direction* (stand at p1
looking at p2: left→right is "forward").

**Events.** Four detectors run on track history only:

| Event | What fires it | Key params |
|---|---|---|
| Loitering | track stays within a small radius too long | `radius_px`, `min_frames` |
| Speeding | smoothed speed exceeds limit | `max_px_per_sec` |
| Wrong-way | motion opposes the declared flow vector | `flow`, `cos_thresh` |
| Crowding | too many tracks inside one zone | `max_tracks`, `min_frames` |

Each detector fires once per track (no spam), and every event carries its
evidence.

## Configuration

Everything lives in `config.yaml` — no code changes needed to retune:

```yaml
detector:
  min_area: 500        # ignore blobs smaller than this (px^2)
  history: 500         # MOG2 background history (frames)
  var_threshold: 16    # MOG2 sensitivity; lower = more sensitive

tracker:
  high_thresh: 0.5     # stage-1 association threshold
  low_thresh: 0.15     # stage-2 rescue threshold
  max_iou_dist: 0.7    # reject matches with cost above this (1 - IoU)
  n_init: 3            # hits before Tentative -> Confirmed
  max_age: 30          # frames before a Tentative track is deleted
  max_lost: 60         # frames before a Lost track is deleted

zones:
  - name: "door"
    polygon: [[40, 200], [280, 200], [280, 440], [40, 440]]

tripwires:
  - name: "gate"
    p1: [160, 60]
    p2: [160, 420]

events:
  loitering: { enabled: true, radius_px: 40, min_frames: 150 }
  speeding:  { enabled: true, max_px_per_sec: 300 }
  wrong_way: { enabled: true, flow: [1, 0] }
  crowding:  { enabled: false, max_tracks: 5 }
```

## Tests

18 tests, all on synthetic data — no camera needed:

```bash
python -m pytest tests/ -q
```

- `test_hungarian.py` — optimality on square/rectangular matrices, ties, empties
- `test_tracker.py` — two targets tracked for 60 frames; **no ID switch**
  through a 5-frame occlusion rescued by weak detections; flicker ghosts never
  confirm; lost tracks are deleted
- `test_zones.py` — zone enter/exit + dwell frames; tripwire direction;
  vanished tracks are forgotten
- `test_events.py` — each behavioral event fires exactly once, and stays
  silent for well-behaved targets

## Project layout

```
src/
  detector.py   motion detector (MOG2) + Detector interface for DNN plug-ins
  yolo_detector.py  YOLO11n via cv2.dnn (see docs/yolo_detector.md)
  kalman.py     single-target Kalman filter, adaptive process noise
  hungarian.py  Kuhn-Munkres from scratch, no SciPy
  tracker.py    multi-object tracker: 3-stage association (IoU + re-ID) + state machine
  appearance.py zero-weight HSV embeddings + re-ID gallery
  multicam.py   multi-camera handover with global identities
  dashboard.py  live web dashboard (stdlib HTTP + SSE)
  zones.py      polygon zones (entry/exit/dwell) + directed tripwires
  events.py     loitering / speeding / wrong-way / crowding detectors
  analytics.py  trajectory CSV recording + session summary + quality check
  main.py       CLI, per-stage timing, live visualization, --dashboard
tests/          18 synthetic tests, no camera required
config.yaml     all thresholds, zones, tripwires, event parameters
```

## Use cases

**Where this wins vs. commercial systems** (Hikvision/Dahua):
zero software cost, runs fully on-premise (video never leaves your network),
and every zone, tripwire and event rule is customizable — no black box.

- **Small retail footfall counting** — a few hundred visitors a day, where a
  commercial people-counting camera is overkill. Tripwire + dwell analysis
  out of the box.
- **Construction site / warehouse intrusion alerts** — few targets, simple
  scene. MOG2 mode needs no GPU, runs on an old PC.
- **Office / server-room loitering detection** — privacy-sensitive areas
  that require local-only processing.
- **Quick prototypes for clients** — demo a working counter in an afternoon
  before committing to hardware.

**Where it doesn't fit**: high-density crowds (subway stations —
ID switches under heavy occlusion), 24/7 high-reliability deployments,
or anything needing face recognition.

## Limitations

- The bundled detector is motion-based: it needs a **static camera** and
  struggles with camouflaged or very slow targets. Plug in a DNN detector
  for the general case.
- No appearance (ReID) features — identity survives short occlusions via
  motion prediction, not long ones via looks. A deliberate trade-off for zero
  weights and CPU-only real-time.
- Pixel-space speeds and distances: calibrate to meters if you need real
  units.

## Roadmap

- [x] YOLOv8/YOLO11 DNN detector plug-in — [`src/yolo_detector.py`](src/yolo_detector.py) · [theory](docs/yolo_detector.md) · [MOG2 vs YOLO](docs/comparison.md)
- [x] Appearance re-identification for long occlusions — [`src/appearance.py`](src/appearance.py) · zero-weight HSV histograms
- [x] Multi-camera handover with global identities — [`src/multicam.py`](src/multicam.py)
- [x] Live web dashboard (counts + event stream) — [`src/dashboard.py`](src/dashboard.py) · stdlib only
- [ ] Multi-camera handoff
- [ ] Web dashboard for live counts and event feed

## Contributing

Bug reports and feature requests are welcome — please use the issue templates
so we get the details needed to reproduce. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the full guide (code style, test requirements, bilingual docs rule).

## License

MIT — see [LICENSE](LICENSE).
