#!/usr/bin/env python3
"""Train the A/B/D groups and the BiFlow ablation variants.

Groups (main results):
  A   YOLO26n @ 640, no BiFlow
  B   YOLO26n @ 1280, no BiFlow
  D   YOLO26n @ 1280 + BiFlow

Ablation variants (all @ 1280): baseline / l2h / h2l / biflow1 / biflow2 /
budget_matched / full. ``full`` is identical to group D.

Training recipe (identical across groups and variants):
  MuSGD, lr0 0.005, cosine schedule, 3 warmup epochs, AMP, 150 epochs,
  patience 30, seed 42, deterministic, workers 0.

Examples:
  python train.py --data data.yaml --weights yolo26n.pt --group D
  python train.py --data data.yaml --weights yolo26n.pt --variant full
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import yolo_improved  # noqa: F401  (registers the custom modules for unpickling)
from ablation import inject_ablation
from yolo_improved import inject_biflow, set_seed

CONFIGS = {
    "A": {"imgsz": 640, "inject": None},
    "B": {"imgsz": 1280, "inject": None},
    "D": {"imgsz": 1280, "inject": "full"},
    "baseline": {"imgsz": 1280, "inject": None},
    "l2h": {"imgsz": 1280, "inject": "l2h"},
    "h2l": {"imgsz": 1280, "inject": "h2l"},
    "biflow1": {"imgsz": 1280, "inject": "biflow1"},
    "biflow2": {"imgsz": 1280, "inject": "biflow2"},
    "budget_matched": {"imgsz": 1280, "inject": "budget_matched"},
    "full": {"imgsz": 1280, "inject": "full"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--weights", default="yolo26n.pt", help="base model checkpoint")
    p.add_argument("--group", choices=["A", "B", "D"], help="main-results group")
    p.add_argument("--variant", choices=sorted(CONFIGS), help="ablation variant")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr0", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs", help="output directory")
    p.add_argument("--no-eval", action="store_true", help="skip test evaluation")
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.group is not None and args.variant is not None:
        raise SystemExit("Use either --group or --variant, not both.")
    if args.group is None and args.variant is None:
        raise SystemExit("Provide --group (A/B/D) or --variant.")
    config_name = args.group if args.group is not None else args.variant
    cfg = CONFIGS[config_name]

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    injector = None
    if cfg["inject"] == "full":
        injector = inject_biflow
    elif cfg["inject"] is not None:
        injector = lambda m: inject_ablation(m, cfg["inject"])  # noqa: E731

    class InjectedTrainer(DetectionTrainer):
        def __init__(self, *a, injector=None, **kw):
            super().__init__(*a, **kw)
            self._injector = injector

        def setup_model(self):
            super().setup_model()
            if self._injector is not None:
                self._injector(self.model)

    set_seed(args.seed)
    model = YOLO(args.weights)
    run_name = f"{config_name}_seed{args.seed}"

    training_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": cfg["imgsz"],
        "batch": args.batch,
        "optimizer": "MuSGD",
        "lr0": args.lr0,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "patience": 30,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "amp": True,
        "device": args.device,
        "workers": 0,
        "cos_lr": True,
        "close_mosaic": 10,
        "pretrained": True,
        "deterministic": True,
        "seed": args.seed,
        "val": True,
        "save": True,
        "save_period": -1,
        "plots": True,
        "cache": False,
        "rect": False,
        "single_cls": False,
        "fraction": 1.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 0.5,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.1,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "max_det": 300,
        "project": args.out,
        "name": run_name,
        "exist_ok": True,
        "trainer": partial(InjectedTrainer, injector=injector),
    }

    model.train(**training_args)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not args.no_eval and best.exists():
        eval_model = YOLO(str(best))
        r = eval_model.val(
            data=args.data, split="test", imgsz=cfg["imgsz"],
            batch=args.batch, device=args.device, workers=0,
            plots=False, verbose=False,
        )
        metrics = {
            "map50": round(float(r.box.map50), 4),
            "map50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
        }
        print(json.dumps({"config": config_name, "test_metrics": metrics}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
