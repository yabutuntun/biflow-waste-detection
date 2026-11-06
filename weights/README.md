# Weights

The paper checkpoints are distributed as a separate release asset,
`weights_v1.0.1.zip`, so the source repository stays code-only. The archive
contains the three models of the controlled comparison, as PyTorch `best.pt`
checkpoints and ONNX exports. SHA256 checksums are listed below and in
`weights_manifest.json`.

The released checkpoints are the leave-one-sequence-out fold-0 models (seed
42), trained on the five clean acquisition sequences (2876 images). The
5-fold mean +/- std reported in the paper is reproduced in
`weights_manifest.json` for reference.

## Released files

| File | Role | Input | SHA256 |
|---|---|---|---|
| `A_640_best.pt` | Baseline A | 640 x 640 | `8ea9d8b6727b93720e5082751dd67cdfd1d4ba4fbad2b97e1c08bdba5a92504d` |
| `B_1280_best.pt` | Baseline B | 1280 x 1280 | `584f957dd9807552ef0d94f9b106d1527a7fcf89084abb86c5bc78708c8f473f` |
| `D_1280_biflow_best.pt` | BiFlow (D) | 1280 x 1280 | `be7a5c2c89e668297973188ca5ed9622d941be7bfcda779ea5bb47e65df63d01` |
| `A_640.onnx` | ONNX export of A | 640 x 640 | `c8840788d04d51dacc738d2a6a42c119e2898bea9dffb22553fbace573e17f21` |
| `B_1280.onnx` | ONNX export of B | 1280 x 1280 | `9354c118c5366d2ab125b636fa4debb54cabfdcad344b66d7ab4b7937bfb5378` |
| `D_1280_biflow.onnx` | ONNX export of D | 1280 x 1280 | `201ea8cf61df83d2613a01361240624b0e214a8fe572324df1dbde80b2f72dad` |

The fusion-control checkpoints reported in the paper are not part of this
archive. They are available from the corresponding author on reasonable
request.

## Layout

```text
weights_v1.0.1/
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
python -c "import hashlib, pathlib; [print(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(pathlib.Path('weights_v1.0.1').glob('*.*'))]"
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
2. Download `weights_v1.0.1.zip` from the release.
3. Unzip the archive and verify the SHA256 checksums as shown above.
