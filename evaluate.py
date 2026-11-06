#!/usr/bin/env python3
"""Evaluate a trained checkpoint: mAP, per-class AP, parameters and GFLOPs.

Example:
  python evaluate.py --weights runs/D_seed42/weights/best.pt --data data.yaml --imgsz 1280
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yolo_improved  # noqa: F401  (registers the custom modules for unpickling)


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_gflops(model, imgsz: int) -> float | None:
    try:
        import torch
        from thop import profile
    except Exception:
        return None

    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(flops / 1e9)


def per_class_metrics(metrics, names: dict) -> dict:
    ap_values = [float(v) for v in metrics.box.ap]
    class_indices = (
        [int(i) for i in metrics.box.ap_class_index]
        if hasattr(metrics.box, "ap_class_index")
        else list(range(len(ap_values)))
    )
    ap = {}
    for position, class_index in enumerate(class_indices):
        class_name = str(names.get(class_index, class_index))
        if position < len(ap_values):
            ap[class_name] = round(ap_values[position], 4)
    return ap


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default=None, help="write JSON result to this path")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, split="test", imgsz=args.imgsz,
        device=args.device, workers=0, verbose=False, plots=False,
    )
    result = {
        "weights": str(Path(args.weights).resolve()),
        "imgsz": args.imgsz,
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class_map50_95": per_class_metrics(metrics, model.names),
        "params": int(parameter_count(model.model)),
        "gflops": profile_gflops(model.model, args.imgsz),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
