# Dataset

The experiments in the paper use the public TOD (Trash Object Detection)
dataset:

```text
https://github.com/jms0923/tod
```

Please cite the original dataset paper:

> Jo, M.-S., Han, S.-S., Jeong, C.-S.: TOD: Trash Object Detection Dataset.
> Journal of Information Processing Systems 18(4), 524-534 (2022).
> https://doi.org/10.3745/JIPS.02.0178

## What is provided here

`../splits/split_manifest.json` records the train/val/test identifiers used in
the paper:

- train: 3972 images
- val: 487 images
- test: 510 images

The manifest contains file identifiers and the original split provenance only.
It does not contain images or labels.

## Expected local layout

After downloading the original TOD data and applying the split manifest, the
training code expects:

```text
korea_waste_grouped_v2/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

The dataset itself is governed by the license and terms of the original TOD
release.
