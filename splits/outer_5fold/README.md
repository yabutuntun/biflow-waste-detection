# Leave-one-sequence-out fold assignments

The paper's main results hold out one acquisition sequence at a time. The
fold assignments are given here so the protocol can be rebuilt from the TOD
release without regenerating splits.

## Files

| File | Content |
|---|---|
| `folds.json` | For each fold, the image identifiers (`stem`, no extension) of `outer_test`, `inner_train` and `inner_val`. |
| `folds_summary.json` | Per-fold image and object counts per class, the source of Supplementary Table S1. |

## Fold to held-out sequence

| Fold | Outer test sequence | Images |
|---|---|---|
| 0 | 20201120 | 300 |
| 1 | 20201207 | 749 |
| 2 | 20201208 | 794 |
| 3 | 20201209 | 633 |
| 4 | GoPro (gopro) | 400 |

Each fold partitions the same 2876 images: `outer_test + inner_train +
inner_val = 2876`, and no identifier appears in two parts. `inner_val` is the
sequence-disjoint validation split used for early stopping and checkpoint
selection; it never overlaps `outer_test` or `inner_train`.

The single train/val/test split used for the component ablation is a different
split and is given in `../split_manifest.json` (train 1943 / val 300 /
test 633). The ablation test sequence 20201209 is also the fold-3 outer test;
the paper states this overlap explicitly.
