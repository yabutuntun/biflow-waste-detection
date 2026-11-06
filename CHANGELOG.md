# Changelog

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
