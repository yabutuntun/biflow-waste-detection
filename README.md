# BiFlow: Lightweight Bidirectional Residual Feature Compensation

Standalone PyTorch implementation of the BiFlow module used in the paper:

> **Accuracy--Efficiency--Latency Tradeoff of High-Resolution Input and
> Lightweight Bidirectional Feature Compensation for Edge Waste Detection**
> Weiru Guo and Thongchai Kaewkiriya, Journal of Real-Time Image Processing
> (submitted).

## Contents

- `biflow.py` -- `BiFlow` and `BiFlowHead`, implemented with PyTorch only.
- `example.py` -- minimal shape example.
- `requirements.txt` -- runtime dependency.
- `pyproject.toml` -- minimal package metadata.
- `tests/test_biflow.py` -- shape tests.
- `data/README.md` -- dataset source and directory layout.
- `splits/split_manifest.json` -- train/val/test identifiers used in the paper.
- `weights/README.md` -- weight availability and checksums.
- `weights/weights_manifest.json` -- SHA256 checksums of the released checkpoints and ONNX exports.
- `LICENSE` -- MIT license for this standalone implementation.
- `NOTICE` -- scope of the release.
- `THIRD_PARTY_NOTICES.md` -- licensing notes for Ultralytics, PyTorch and the TOD dataset.
- `CHANGELOG.md` -- release history.
- `CITATION.cff` -- citation metadata.

The source repository does not bundle the Ultralytics integration layer, the
full training/evaluation pipeline, the TOD dataset, or trained weights. The
checkpoints and ONNX exports are published as a separate release asset; see
`weights/README.md`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from biflow import BiFlow, BiFlowHead

f_high = torch.randn(1, 64, 80, 80)
f_low = torch.randn(1, 128, 40, 40)

pair = BiFlow(c_high=64, c_low=128)
y_high, y_low = pair(f_high, f_low)

p3 = torch.randn(1, 64, 80, 80)
p4 = torch.randn(1, 128, 40, 40)
p5 = torch.randn(1, 256, 20, 20)

head = BiFlowHead(c3=64, c4=128, c5=256)
p3_out, p4_out, p5_out = head([p3, p4, p5])
```

Run the complete example:

```bash
python example.py
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

`BiFlowHead` connects two units across P3/P4/P5. P4 is compensated twice,
first with P3 and then with P5.

## Dataset

The experiments use the public TOD (Trash Object Detection) dataset:

```text
https://github.com/jms0923/tod
```

The dataset is not redistributed in this repository. The exact train/val/test
identifiers used in the paper are provided in `splits/split_manifest.json`;
images and labels must be obtained from the original TOD release. See
`data/README.md` for the expected directory layout and citation.

## Weights

Trained checkpoints and ONNX exports are published as a separate release asset
(`weights_v1.0.0.zip`), so the source repository stays code-only. The archive
contains the three models of the controlled comparison together with a checksum
manifest; `weights/README.md` lists the released files and
`weights/weights_manifest.json` records their SHA256 checksums. The checkpoints
were produced with the Ultralytics framework, and their use remains subject to
the corresponding Ultralytics license terms.

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
