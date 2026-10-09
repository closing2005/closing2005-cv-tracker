# Contributing to cv-tracker

Thanks for considering a contribution. This is a small, focused project — the
best contributions are tight, tested, and explainable.

## How to contribute

1. Fork the repo and create a branch: `git checkout -b feat/my-idea`
2. Add tests under `tests/` for any behavior change. Run them:
   `python -m pytest tests/ -q` — all 18+ must pass.
3. Keep the dependency footprint tiny: `opencv-python`, `numpy`, `pyyaml`.
   Anything heavier needs a strong reason.
4. Update `README.md` **and** `README.zh-CN.md` (written natively in each
   language, not translated sentence-by-sentence), plus `GLOSSARY.md` if you
   introduce new terminology.
5. Open a pull request with a clear description of *what* and *why*.

## Code style

- Plain, readable Python. No clever one-liners that need a paragraph to explain.
- Every magic number gets a comment or moves to `config.yaml`.
- New detectors subclass `Detector` in `src/detector.py` — never touch the
  tracker to add a detector.

## Reporting issues

Use the issue templates (bug report / feature request) so we get the details
needed to reproduce: source type, `config.yaml` changes, and the session
summary output.
