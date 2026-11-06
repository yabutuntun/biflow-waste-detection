#!/usr/bin/env python3
"""Build the clean_v1 splits (2876 images, 5 acquisition sequences).

Generates:
  - a single ablation split (train / val / test), and
  - a leave-one-sequence-out 5-fold split.

The five clean sequences are the only ones used in the paper. Each line of the
produced txt files is an absolute image path, matching the Ultralytics format.

Example:
  python make_splits.py --images korea_waste_grouped_v2/images \
      --labels korea_waste_grouped_v2/labels --out splits
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

CLASSES = ["paper", "paperpack", "papercup", "can", "bottle",
           "pet", "plastic", "vinyl", "cap", "label"]

SEQS = {
    "20201120": r"^20201120_",
    "20201207": r"^20201207_",
    "20201208": r"^20201208_",
    "20201209": r"^20201209_",
    "gopro": r"^(GOPR|GP|GX|GH)\d",
}


def collect(images: Path, labels: Path) -> dict:
    seq_imgs = defaultdict(list)
    for split in ("train", "val", "test"):
        d = images / split
        if not d.exists():
            continue
        for fn in sorted(os.listdir(d)):
            stem = os.path.splitext(fn)[0]
            for seq, pat in SEQS.items():
                if re.match(pat, stem):
                    seq_imgs[seq].append((split, stem, str(d / fn)))
                    break

    result = {}
    for seq, items in seq_imgs.items():
        counts = defaultdict(int)
        for split, stem, _ in items:
            lbl = labels / split / (stem + ".txt")
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    a = line.split()
                    if a:
                        counts[int(float(a[0]))] += 1
        result[seq] = {
            "items": items,
            "counts": counts,
            "n_img": len(items),
            "n_cls": sum(1 for i in range(10) if counts[i] > 0),
        }
    return result


def counts_of(items, labels: Path) -> dict:
    c = defaultdict(int)
    for split, stem, _ in items:
        lbl = labels / split / (stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                a = line.split()
                if a:
                    c[int(float(a[0]))] += 1
    return {str(k): v for k, v in sorted(c.items())}


def write_txt(path: Path, items):
    path.write_text("\n".join(p for _, _, p in items) + "\n", encoding="utf-8")


def write_yaml(path: Path, dataset_root: Path, train_txt, val_txt, test_txt):
    lines = [
        f"path: {dataset_root.as_posix()}",
        f"train: {train_txt}",
        f"val: {val_txt}",
        f"test: {test_txt}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="dataset images root (train/val/test)")
    p.add_argument("--labels", required=True, help="dataset labels root (train/val/test)")
    p.add_argument("--out", default="splits", help="output directory")
    args = p.parse_args()

    images = Path(args.images)
    labels = Path(args.labels)
    out = Path(args.out)

    data = collect(images, labels)
    total = sum(data[s]["n_img"] for s in SEQS)
    assert total == 2876, f"total image count {total} != 2876"
    print(f"total images: {total}")

    # 1) single ablation split: test=20201209, val=20201120, train=1207+1208+gopro
    ab_out = out / "ablation"
    ab_out.mkdir(parents=True, exist_ok=True)
    train_seqs = ["20201207", "20201208", "gopro"]
    val_seq = "20201120"
    test_seq = "20201209"

    train_items = [it for s in train_seqs for it in data[s]["items"]]
    val_items = data[val_seq]["items"]
    test_items = data[test_seq]["items"]
    write_txt(ab_out / "train.txt", train_items)
    write_txt(ab_out / "val.txt", val_items)
    write_txt(ab_out / "test.txt", test_items)
    write_yaml(ab_out / "data.yaml", images.parent,
               str(ab_out / "train.txt"), str(ab_out / "val.txt"),
               str(ab_out / "test.txt"))
    print(f"ablation: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    # 2) leave-one-sequence-out 5-fold split
    nc_out = out / "outer_5fold"
    nc_out.mkdir(parents=True, exist_ok=True)
    seq_order = list(SEQS.keys())
    folds_summary = {"outer_folds": []}

    for fi, test_seq in enumerate(seq_order):
        fold_dir = nc_out / f"fold_{fi}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        outer_test_items = data[test_seq]["items"]
        remaining = [s for s in seq_order if s != test_seq]

        # inner_val: prefer class coverage (10 classes), then image count nearest the mean
        mean_n = sum(data[s]["n_img"] for s in remaining) / len(remaining)
        inner_val_seq = max(
            remaining,
            key=lambda s: (data[s]["n_cls"], -abs(data[s]["n_img"] - mean_n)))
        inner_train_seqs = [s for s in remaining if s != inner_val_seq]

        inner_train_items = [it for s in inner_train_seqs for it in data[s]["items"]]
        inner_val_items = data[inner_val_seq]["items"]

        write_txt(fold_dir / "inner_train.txt", inner_train_items)
        write_txt(fold_dir / "inner_val.txt", inner_val_items)
        write_txt(fold_dir / "outer_test.txt", outer_test_items)
        write_yaml(fold_dir / "data.yaml", images.parent,
                   str(fold_dir / "inner_train.txt"),
                   str(fold_dir / "inner_val.txt"),
                   str(fold_dir / "outer_test.txt"))

        folds_summary["outer_folds"].append({
            "fold": fi,
            "outer_test": {"images": len(outer_test_items), "seq": test_seq,
                           "class_counts": counts_of(outer_test_items, labels)},
            "inner_val": {"images": len(inner_val_items), "seq": inner_val_seq,
                          "class_counts": counts_of(inner_val_items, labels)},
        })
        print(f"fold_{fi}: test={test_seq}({len(outer_test_items)}) "
              f"val={inner_val_seq}({len(inner_val_items)}) "
              f"train={inner_train_seqs}({len(inner_train_items)})")

    (nc_out / "folds_summary.json").write_text(
        json.dumps(folds_summary, indent=2), encoding="utf-8")
    print(f"splits written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
