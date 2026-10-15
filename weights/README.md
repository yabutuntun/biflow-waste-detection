# Weights

The paper checkpoints are distributed as a separate release asset,
`weights_v1.0.0.zip`, so the source repository stays code-only. The archive
contains the three models of the controlled comparison, as PyTorch `best.pt`
checkpoints and ONNX exports. SHA256 checksums are listed below and in
`weights_manifest.json` (a copy of `weights/weights_manifest.json` from the
source repository).

## Released files

| File | Role | Input | SHA256 |
|---|---|---|---|
| `A_640_best.pt` | Baseline A | 640 x 640 | `0e23b05f6549a8ff84c69cfbbc226097a1481c4b6003a6c0f1572ea6c13587de` |
| `B_1280_best.pt` | Baseline B | 1280 x 1280 | `73a1f77d053911cbacb97fe9755ac2033576fc28f46a4a33dd40e47f48194d4d` |
| `D_1280_biflow_best.pt` | BiFlow (D) | 1280 x 1280 | `851e15e871e78d06da6610cada45b7ea59aeb8ee3d941435ecec09f0aa65abad` |
| `A_640.onnx` | ONNX export of A | 640 x 640 | `6090668bdb97b360587474a5c4b117e3114bb68cb4140f7f3e890b7c05bfa214` |
| `B_1280.onnx` | ONNX export of B | 1280 x 1280 | `9d0044ba67465865cc4927fea809fcfbb981bd53a39a203803c65ad9129f0e8d` |
| `D_1280_biflow.onnx` | ONNX export of D | 1280 x 1280 | `670b4c2ca103f9293e67461c24c24f86cd7d34f1b3dfbd7ec99222398d3c30e9` |

The two fusion-control checkpoints reported in the paper are not part of this
archive. They are available from the corresponding author on reasonable
request.

## Layout

```text
weights_v1.0.0/
  A_640_best.pt
  B_1280_best.pt
  D_1280_biflow_best.pt
  A_640.onnx
  B_1280.onnx
  D_1280_biflow.onnx
  weights_manifest.json
  README.md
```

## Verification

```bash
python -c "import hashlib, pathlib; [print(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(pathlib.Path('weights_v1.0.0').glob('*.*'))]"
```

Compare the printed digests with `weights_manifest.json` before use.

## Licensing

The checkpoints and ONNX exports were produced with the Ultralytics framework.
The baseline detector is built on Ultralytics YOLO26n, and the Ultralytics
license terms apply to that framework; see `THIRD_PARTY_NOTICES.md` in the
source repository. The standalone `biflow.py` in the source repository is
MIT-licensed and is independent of Ultralytics.

## Download

The archive is attached to the GitHub release page of the repository:

1. Open the Releases page of the repository.
2. Download `weights_v1.0.0.zip` from the latest release.
3. Unzip the archive and verify the SHA256 checksums as shown above.
