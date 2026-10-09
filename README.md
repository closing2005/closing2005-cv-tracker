# cv-tracker

Real-time multi-object tracking for static cameras. Detection → Kalman-filtered
tracking → zones/tripwires → behavioral events, in one CLI with zero weight
files and zero SciPy.

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
  detections. Occluded targets are rescued instead of reborn with new IDs.
- **Kalman filter, constant-velocity model**, with process noise scaled by
  target size — large boxes tolerate more motion uncertainty without per-scene
  tuning.
- **Hungarian algorithm implemented from scratch** (`src/hungarian.py`,
  O(n³), rectangular matrices). No SciPy, no black box.
- **Track lifecycle that kills flicker ghosts**: `Tentative → Confirmed →
  Lost → Deleted`. Only confirmed tracks are reported or counted.
- **Behavioral events, not just boxes**: loitering, speeding, wrong-way
  motion, crowding — computed from track history alone, every event carrying
  its evidence (track id, frame, measured value).
- **Tracker self-diagnostics.** The session summary reports fragmentation
  rate and a quality verdict, so you know whether to trust the numbers.
- **Pluggable detectors.** Ships with a weight-free MOG2 motion detector;
  subclass `Detector` to plug in YOLO via `cv2.dnn` without touching the
  tracker.

## Quickstart

```bash
pip install -r requirements.txt

# webcam
python -m src.main --source 0 --show

# video file, headless, save trajectories
python -m src.main --source traffic.mp4 --no-show --out runs/run1 --max-frames 900
```

Outputs in `--out/`: `trajectories.csv` (frame, track_id, box, center,
velocity) and `summary.yaml` (counts, speeds, events, track-quality verdict).

## Configuration

All thresholds live in `config.yaml`: detector sensitivity, tracker
association gates, zone polygons, tripwire segments, and per-event parameters
(loitering radius/frames, speed limit, flow direction, crowd limit).

## Tests

```bash
python -m pytest tests/ -q
```

18 tests, all synthetic (no camera needed): Hungarian optimality on square
and rectangular matrices, ID stability through occlusion with weak-detection
rescue, ghost filtering, zone enter/exit + dwell, tripwire direction,
and every behavioral event firing exactly once.

## Project layout

```
src/
  detector.py   motion detector (MOG2) + Detector interface for DNN plug-ins
  kalman.py     single-target Kalman filter, adaptive process noise
  hungarian.py  Kuhn-Munkres from scratch, no SciPy
  tracker.py    multi-object tracker: 2-stage association + track state machine
  zones.py      polygon zones (entry/exit/dwell) + directed tripwires
  events.py     loitering / speeding / wrong-way / crowding detectors
  analytics.py  trajectory CSV recording + session summary + quality check
  main.py       CLI, per-stage timing, live visualization
```

## Limitations (honest)

- The bundled detector is motion-based: it needs a **static camera** and
  struggles with camouflaged or very slow targets. Plug in a DNN detector
  for the general case.
- No appearance (ReID) features — identity survives short occlusions via
  motion prediction, not long ones via looks. That's a deliberate trade-off
  for zero weights and CPU-only real-time.
- Pixel-space speeds/distances: calibrate to meters if you need real units.

## License

MIT
