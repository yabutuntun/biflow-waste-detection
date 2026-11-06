#!/usr/bin/env python3
"""Export trained checkpoints to ONNX (FP32, NMS-free head).

The BiFlow checkpoints need the custom modules registered before loading, so
this script imports yolo_improved first. Export with the same Ultralytics
version that produced the checkpoint (the paper uses 8.4.106).

Examples:
  python export.py --weights runs/D_seed42/weights/best.pt --imgsz 1280 --out onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yolo_improved  # noqa: F401  (registers the custom modules for unpickling)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--out", default="onnx", help="output directory")
    args = p.parse_args()

    from ultralytics import YOLO

    pt = Path(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(pt))
    model.export(format="onnx", imgsz=args.imgsz, dynamic=False)

    src = pt.with_suffix(".onnx")
    dst = out_dir / f"{pt.stem}_{args.imgsz}.onnx"
    if src.exists():
        shutil.move(str(src), str(dst))
    else:
        raise FileNotFoundError(f"Export did not produce {src}")

    manifest = {"onnx": str(dst.resolve()), "src_pt": str(pt.resolve()), "imgsz": args.imgsz}
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
