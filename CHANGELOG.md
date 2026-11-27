# Changelog

## v1.0.3 (2026-11-28)

- Bumped the metadata version in `CITATION.cff` and `pyproject.toml`; the
  release archives carried the previous version string.
- Renamed the weights asset to `weights_v1.0.3.zip` (contents unchanged, same
  SHA256 values; only the enclosing folder is renamed), and updated the
  references in both READMEs and in `weights/weights_manifest.json`.
- `manifest.json` regenerated.

## v1.0.2 (2026-11-28)

- Added `splits/outer_5fold/` with the leave-one-sequence-out fold assignments
  used for the main results, and the per-fold class counts behind
  Supplementary Table S1.
- Added `measurements/` with the raw Raspberry Pi 5 latency, power and peak
  memory output behind the timing tables.
- Linked the Zenodo archive in the README.

## v1.0.1 (2026-11-06)

- Upgraded the repository to the full reproduction pipeline.
- Added `yolo_improved.py` (Ultralytics integration: BiFlow, BiFlowDetect,
  inject_biflow, plus the BiFPN / TridentNet / focal-loss controls).
- Added `ablation.py`, `train.py`, `evaluate.py`, `export.py` and
  `make_splits.py`.
- Updated the split manifest to the clean_v1 split (2876 images, five
  acquisition sequences; train 1943 / val 300 / test 633).
- Updated the weights manifest to the leave-one-sequence-out fold-0 A/B/D
  checkpoints and their ONNX exports.
- Cleaned up module comments and docstrings.

## v1.0.0 (2026-10-15)

- Initial public release of the standalone BiFlow module.
- Added shape tests and a minimal usage example.
- Added the train/val/test split manifest used in the paper.
- Added the checkpoint and ONNX checksum manifest for the release assets.
