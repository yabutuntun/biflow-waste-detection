# BiFlow: Lightweight Bidirectional Residual Feature Compensation

Code for the paper:

> **Accuracy--Efficiency--Latency Tradeoff of High-Resolution Input and
> Lightweight Bidirectional Feature Compensation for Edge Waste Detection**
> Weiru Guo and Thongchai Kaewkiriya, Journal of Real-Time Image Processing
> (submitted).

BiFlow is a lightweight bidirectional feature-compensation module inserted
between the feature pyramid and the detection head. It fuses each pair of
adjacent feature levels (P3/P4 and P4/P5) with two residual branches: a
low-to-high (upsampling) branch and a high-to-low (downsampling) branch. On the
TOD waste-sorting dataset, BiFlow improves the leave-one-sequence-out
mAP@0.50:0.95 of the YOLO26n @ 1280 baseline by +0.020 at a cost of +8.4%
FLOPs.

## Contents

- `biflow.py` -- standalone `BiFlow` and `BiFlowHead`, PyTorch only.
- `yolo_improved.py` -- the modules integrated with Ultralytics (`BiFlow`,
  `BiFlowDetect`, `inject_biflow`, plus the BiFPN / TridentNet / focal-loss
  controls used in the paper).
- `ablation.py` -- the ablation variants (l2h / h2l / biflow1 / biflow2 /
  budget_matched).
- `train.py` -- training script for groups A/B/D and the ablation variants.
- `evaluate.py` -- mAP, per-class AP, parameters and GFLOPs evaluation.
- `export.py` -- ONNX export (FP32, NMS-free head).
- `make_splits.py` -- build the clean_v1 splits from the TOD dataset.
- `example.py` -- minimal shape example.
- `splits/split_manifest.json` -- train/val/test identifiers used in the paper.
- `weights/weights_manifest.json` -- SHA256 checksums of the released
  checkpoints and ONNX exports.
- `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` -- licensing.

The TOD dataset and the trained weights are not bundled in this repository.
The checkpoints and ONNX exports are published as a separate release asset;
see `weights/README.md`.

## Installation

```bash
pip install -r requirements.txt
```

The full pipeline needs a GPU and the Ultralytics framework (>= 8.3.0). The
standalone `biflow.py` only needs PyTorch.

## Usage

### Standalone module

```python
import torch
from biflow import BiFlow, BiFlowHead

f_high = torch.randn(1, 64, 80, 80)
f_low = torch.randn(1, 128, 40, 40)

pair = BiFlow(c_high=64, c_low=128)
y_high, y_low = pair(f_high, f_low)
```

### Training

Download the YOLO26n pretrained weights (`yolo26n.pt`) and a TOD data.yaml,
then run:

```bash
# main-results groups
python train.py --data data.yaml --weights yolo26n.pt --group A   # 640 baseline
python train.py --data data.yaml --weights yolo26n.pt --group B   # 1280 baseline
python train.py --data data.yaml --weights yolo26n.pt --group D   # 1280 + BiFlow

# ablation variants
python train.py --data data.yaml --weights yolo26n.pt --variant full
python train.py --data data.yaml --weights yolo26n.pt --variant l2h
```

The training recipe is identical across configurations: MuSGD, lr0 0.005,
cosine schedule, 3 warmup epochs, AMP, 150 epochs, patience 30, seed 42,
deterministic, workers 0.

### Evaluation

```bash
python evaluate.py --weights runs/D_seed42/weights/best.pt --data data.yaml --imgsz 1280
```

### ONNX export

```bash
python export.py --weights runs/D_seed42/weights/best.pt --imgsz 1280 --out onnx
```

### Building the splits

```bash
python make_splits.py --images korea_waste_grouped_v2/images \
    --labels korea_waste_grouped_v2/labels --out splits
```

## Method

For two adjacent feature levels, the module applies two residual branches:

```text
y_high = f_high + low_to_high(f_low)
y_low  = f_low  + high_to_low(f_high)
```

The low-to-high branch uses bilinear upsampling followed by two 1x1
convolutions. The high-to-low branch uses a stride-2 3x3 convolution followed
by a 1x1 convolution. The intermediate width is

```text
m = max((c_high + c_low) // 4, 32)
```

Two units are connected across P3/P4/P5. P4 is compensated twice, first with
P3 and then with P5.

## Dataset

The experiments use the public TOD (Trash Object Detection) dataset:

```text
https://github.com/jms0923/tod
```

The dataset is not redistributed in this repository. The evaluation uses 2876
images grouped into five acquisition sequences (train 1943 / val 300 /
test 633). The exact identifiers are provided in
`splits/split_manifest.json`; images and labels must be obtained from the
original TOD release. See `data/README.md` for the expected layout.

## Weights

Trained checkpoints and ONNX exports are published as a separate release
asset (`weights_v1.0.1.zip`). The archive contains the three models of the
controlled comparison (A/B/D) together with a checksum manifest;
`weights/README.md` lists the released files and
`weights/weights_manifest.json` records their SHA256 checksums. The
checkpoints were produced with the Ultralytics framework, and their use
remains subject to the corresponding Ultralytics license terms.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

## Citation

The paper is under review. Until the final bibliographic record is available,
please cite this repository using `CITATION.cff`.

## License

The standalone implementation in this repository is released under the MIT
License. See `LICENSE` and `NOTICE`.

## Contact

Corresponding author: Thongchai Kaewkiriya
Faculty of Engineering, Shinawatra University, Thailand
Email: thongchai.k@siu.ac.th
