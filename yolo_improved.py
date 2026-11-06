"""YOLO26n improvement modules.

Three modules that integrate into the forward pass or loss computation:

1. BiFlow        -- bidirectional feature flow, applied before the Detect
                    head to fuse the P3/P4/P5 feature levels.
2. TridentBlock  -- TridentNet (Li et al., ICCV 2019): three shared-weight
                    parallel branches with dilation 1/2/3.
3. FocalBCE      -- Focal loss (Lin et al., ICCV 2017) replacing the
                    classification BCE.

Dependencies: torch, ultralytics.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect


# Random seed
def set_seed(seed: int = 42):
    """Fix all random sources for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# BiFlow: bidirectional feature flow
class BiFlow(nn.Module):
    """Bidirectional compensation between two adjacent feature levels.

    forward(f_high, f_low) -> (enhanced f_high, enhanced f_low)
      f_high: higher-resolution feature (c_high channels, e.g. P3=64)
      f_low : lower-resolution feature (c_low  channels, e.g. P4=128)
    """

    def __init__(self, c_high: int, c_low: int):
        super().__init__()
        mid = max((c_high + c_low) // 4, 32)

        # Top-down branch: high -> low resolution (stride-2 downsample).
        self.td = nn.Sequential(
            nn.Conv2d(c_high, mid, 3, 2, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, c_low, 1, bias=False),
        )

        # Bottom-up branch: low -> high resolution (x2 upsample).
        self.bu = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_low, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, c_high, 1, bias=False),
        )

    def forward(self, f_high, f_low):
        return f_high + self.bu(f_low), f_low + self.td(f_high)


class BiFlowDetect(Detect):
    """Detect head with BiFlow: fuse P3/P4/P5 before the original Detect logic.

    forward is defined as a class method (rather than a closure) so the model
    can be serialized by torch.save.
    """

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p3_f, p4_e = self.biflow1(p3, p4)
        p4_f, p5_f = self.biflow2(p4_e, p5)
        return super().forward([p3_f, p4_f, p5_f])


def inject_biflow(detection_model):
    """Attach BiFlow to the Detect head and fuse P3/P4/P5. Returns the modules."""
    detect = detection_model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(detect).__name__}")

    chs = [detect.cv2[i][0].conv.in_channels for i in range(3)]
    print(f"[inject_biflow] P3/P4/P5 channels = {chs}")

    biflow1 = BiFlow(chs[0], chs[1])  # P3 <-> P4
    biflow2 = BiFlow(chs[1], chs[2])  # P4 <-> P5
    detect.add_module("biflow1", biflow1)
    detect.add_module("biflow2", biflow2)

    # Switch the existing Detect instance to the BiFlow subclass, keeping all
    # existing state and parameters.
    detect.__class__ = BiFlowDetect
    return biflow1, biflow2


# BiFPN adaptations
# The canonical variant replaces the existing FPN/PAN neck with a three-level,
# three-repeat BiFPN adaptation. The lite variant keeps the BiFlow insertion
# point and branch widths, but replaces simple residual additions with
# normalized weighted bidirectional fusion.


class FastNormalizedFusion(nn.Module):
    """Scalar weighted fusion with Fast Normalized Fusion."""

    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        if num_inputs < 2:
            raise ValueError("FastNormalizedFusion requires at least two inputs")
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.eps = eps

    def forward(self, inputs):
        if len(inputs) != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} inputs, received {len(inputs)}"
            )
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        out = weights[0] * inputs[0]
        for weight, feature in zip(weights[1:], inputs[1:]):
            out = out + weight * feature
        return out


def _resize_like(feature, reference):
    """Nearest-neighbor resize to the spatial size of the reference feature."""
    if feature.shape[-2:] != reference.shape[-2:]:
        feature = F.interpolate(
            feature, size=reference.shape[-2:], mode="nearest"
        )
    return feature


class SeparableConvBNSiLU(nn.Module):
    """Depthwise separable convolution used by the BiFPN adaptation."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, 1, 1, groups=channels, bias=False
        )
        self.bn_depth = nn.BatchNorm2d(channels)
        self.act_depth = nn.SiLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_point = nn.BatchNorm2d(channels)
        self.act_point = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act_depth(self.bn_depth(self.depthwise(x)))
        return self.act_point(self.bn_point(self.pointwise(x)))


class BiFPNNode(nn.Module):
    """One weighted-fusion node with a separable output convolution."""

    def __init__(self, num_inputs: int, channels: int):
        super().__init__()
        self.fusion = FastNormalizedFusion(num_inputs)
        self.conv = SeparableConvBNSiLU(channels)

    def forward(self, inputs):
        return self.conv(self.fusion(inputs))


class BiFPNBlock(nn.Module):
    """One three-level BiFPN layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.p4_td = BiFPNNode(2, channels)
        self.p3_out = BiFPNNode(2, channels)
        self.p4_out = BiFPNNode(3, channels)
        self.p5_out = BiFPNNode(2, channels)

    def forward(self, p3, p4, p5):
        p4_td = self.p4_td([p4, _resize_like(p5, p4)])
        p3_out = self.p3_out([p3, _resize_like(p4_td, p3)])
        p4_out = self.p4_out(
            [p4, p4_td, _resize_like(p3_out, p4)]
        )
        p5_out = self.p5_out([p5, _resize_like(p4_out, p5)])
        return p3_out, p4_out, p5_out


class BiFPNStack(nn.Module):
    """Project inputs to a common width, repeat BiFPN layers, project outputs."""

    def __init__(
        self,
        c3: int,
        c4: int,
        c5: int,
        common_channels: int = 64,
        repeats: int = 3,
        out_channels=(64, 128, 256),
    ):
        super().__init__()
        if repeats < 1:
            raise ValueError("repeats must be at least 1")

        input_channels = (c3, c4, c5)
        self.input_convs = nn.ModuleList(
            [
                nn.Identity()
                if c == common_channels
                else nn.Conv2d(c, common_channels, 1, bias=False)
                for c in input_channels
            ]
        )
        self.blocks = nn.ModuleList(
            [BiFPNBlock(common_channels) for _ in range(repeats)]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Identity()
                if c == common_channels
                else nn.Conv2d(common_channels, c, 1, bias=False)
                for c in out_channels
            ]
        )

    def forward(self, features):
        if len(features) != 3:
            raise ValueError("BiFPNStack expects exactly three feature levels")
        p3, p4, p5 = [
            conv(feature)
            for conv, feature in zip(self.input_convs, features)
        ]
        for block in self.blocks:
            p3, p4, p5 = block(p3, p4, p5)
        return [
            conv(feature)
            for conv, feature in zip(
                self.output_convs, (p3, p4, p5)
            )
        ]


def _detect_input_channels(detect):
    if not isinstance(detect, Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(detect).__name__}")
    return [detect.cv2[i][0].conv.in_channels for i in range(3)]


def _layer_channels(detection_model, indices, probe_size: int = 128):
    """Infer channel counts at selected layer outputs with a small probe."""
    m = detection_model.model
    device = next(detection_model.parameters()).device
    x = torch.zeros(1, 3, probe_size, probe_size, device=device)
    outputs = []
    channels = {}
    max_index = max(indices)

    with torch.no_grad():
        for i, layer in enumerate(m):
            f = getattr(layer, "f", -1)
            if f != -1:
                x = (
                    outputs[f]
                    if isinstance(f, int)
                    else [outputs[j] if j != -1 else x for j in f]
                )
            x = layer(x)
            outputs.append(x)
            if i in indices:
                if not isinstance(x, torch.Tensor) or x.dim() != 4:
                    raise RuntimeError(
                        f"Layer {i} did not produce a four-dimensional tensor"
                    )
                channels[i] = x.shape[1]
            if i >= max_index:
                break

    missing = [i for i in indices if i not in channels]
    if missing:
        raise RuntimeError(f"Could not infer channels for layers: {missing}")
    return [channels[i] for i in indices]


def inject_bifpn_neck(
    detection_model,
    common_channels: int = 64,
    repeats: int = 3,
    input_layers=(4, 6, 10),
    out_channels=(64, 128, 256),
):
    """Replace the original FPN/PAN neck with a canonical BiFPN adaptation.

    YOLO26n P3/P4/P5 are taken from the backbone outputs at layers 4, 6 and
    10. The original neck layers are removed, while the backbone and the
    one2one NMS-free detection head are retained.
    """
    m = detection_model.model
    layers = list(m.children())
    if len(layers) < max(input_layers) + 1:
        raise RuntimeError("Detection model is shorter than the expected layers")
    if not isinstance(layers[-1], Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(layers[-1]).__name__}")

    c3, c4, c5 = _layer_channels(detection_model, input_layers)
    neck = BiFPNStack(
        c3,
        c4,
        c5,
        common_channels=common_channels,
        repeats=repeats,
        out_channels=out_channels,
    )
    detect = layers[-1]
    prefix = layers[: max(input_layers) + 1]
    new_model = nn.Sequential(*prefix, neck, detect)

    for i, layer in enumerate(new_model):
        layer.i = i
    neck.f = list(input_layers)
    detect.f = len(prefix)

    detection_model.model = new_model
    detection_model.save = list(range(len(new_model)))
    detection_model.yaml = dict(detection_model.yaml)
    detection_model.yaml["head"] = [
        [
            list(input_layers),
            1,
            "BiFPNStack",
            [common_channels, repeats, list(out_channels)],
        ],
        [len(prefix), 1, "Detect", ["nc"]],
    ]
    print(
        "[inject_bifpn_neck] "
        f"in={c3}/{c4}/{c5}, out={out_channels}, "
        f"common={common_channels}, repeats={repeats}"
    )
    return neck


class WeightedBiFlowUnit(nn.Module):
    """Budget-matched weighted variant of one BiFlow unit."""

    def __init__(self, c_high: int, c_low: int):
        super().__init__()
        mid = max((c_high + c_low) // 4, 32)
        self.high_to_low = nn.Sequential(
            nn.Conv2d(c_high, mid, 3, 2, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c_low, 1, bias=False),
        )
        self.low_to_high = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_low, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c_high, 1, bias=False),
        )
        self.high_fusion = FastNormalizedFusion(2)
        self.low_fusion = FastNormalizedFusion(2)

    def forward(self, f_high, f_low):
        high = self.high_fusion([f_high, self.low_to_high(f_low)])
        low = self.low_fusion([f_low, self.high_to_low(f_high)])
        return high, low


class BiFPNLiteDetect(Detect):
    """Head-side weighted BiFPN-lite with the BiFlow insertion point."""

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p3_f, p4_e = self.bifpn_lite1(p3, p4)
        p4_f, p5_f = self.bifpn_lite2(p4_e, p5)
        return super().forward([p3_f, p4_f, p5_f])


def inject_bifpn_lite(detection_model):
    """Attach a budget-matched weighted BiFPN-lite module to Detect."""
    detect = detection_model.model[-1]
    chs = _detect_input_channels(detect)
    print(f"[inject_bifpn_lite] P3/P4/P5 channels = {chs}")

    bifpn_lite1 = WeightedBiFlowUnit(chs[0], chs[1])
    bifpn_lite2 = WeightedBiFlowUnit(chs[1], chs[2])
    detect.add_module("bifpn_lite1", bifpn_lite1)
    detect.add_module("bifpn_lite2", bifpn_lite2)
    detect.__class__ = BiFPNLiteDetect
    return bifpn_lite1, bifpn_lite2


# TridentBlock: TridentNet multi-branch block
class TridentBlock(nn.Module):
    """TridentNet block (Li et al., ICCV 2019).

    Three parallel branches share the same 3x3 convolution weights and differ
    only in dilation (1/2/3), then concat + 1x1 channel refinement + residual.
    Weight sharing points the later branches at the first branch's Parameter.
    A bottleneck ratio keeps the parameter count comparable to the C3k2 block
    it replaces.
    """

    def __init__(self, c1: int, c2: int, dilations=(1, 2, 3), ratio: float = 0.5):
        super().__init__()
        self.dilations = dilations
        mid = max(int(c2 * ratio), 32)
        self.cv_in = nn.Conv2d(c1, mid, 1, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(mid)
        # Three parallel branches with different dilation.
        self.branches = nn.ModuleList(
            [nn.Conv2d(mid, mid, 3, 1, padding=d, dilation=d, bias=False) for d in dilations]
        )
        # Shared weights: the later branches reuse the first branch's weight.
        for b in self.branches[1:]:
            b.weight = self.branches[0].weight
        self.fuse = nn.Conv2d(mid * len(dilations), c2, 1, 1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act(self.bn_in(self.cv_in(x)))
        feats = [branch(out) for branch in self.branches]
        out = torch.cat(feats, dim=1)
        out = self.act(self.bn_fuse(self.fuse(out)))
        if identity.shape[1] == out.shape[1]:
            out = out + identity
        return out


def _layer_io_channels(detection_model, idx):
    """Infer input/output channels of the idx-th Sequential layer via a dummy forward."""
    m = detection_model.model
    dev = next(detection_model.parameters()).device
    x = torch.zeros(1, 3, 64, 64, device=dev)
    y = []
    with torch.no_grad():
        for i, layer in enumerate(m):
            f = getattr(layer, "f", -1)
            if f != -1:
                x = y[f] if isinstance(f, int) else [y[j] if j != -1 else x for j in f]
            in_ch = x.shape[1] if (isinstance(x, torch.Tensor) and x.dim() == 4) else None
            x = layer(x)
            y.append(x)
            if i == idx:
                out_ch = x.shape[1] if x.dim() == 4 else None
                return in_ch, out_ch
    return None, None


def inject_trident(detection_model, layer_idx=8):
    """Replace the deepest backbone C3k2 (default layer 8, 256->256) with TridentBlock.

    Leaves SPPF (layer 9) and C2PSA (layer 10) intact.
    """
    m = detection_model.model
    layer = m[layer_idx]
    if type(layer).__name__ != "C3k2":
        raise RuntimeError(f"Layer {layer_idx} is not C3k2: {type(layer).__name__}")

    c1, c2 = _layer_io_channels(detection_model, layer_idx)
    print(f"[inject_trident] replacing layer {layer_idx} C3k2: {c1} -> {c2} channels")

    block = TridentBlock(c1, c2)
    # Add the layer attributes required by the Ultralytics forward loop.
    block.f = -1
    block.i = layer_idx
    block.type = "TridentBlock"
    m[layer_idx] = block
    return block


# FocalBCE: focal loss
class FocalBCE(nn.Module):
    """Focal loss (Lin et al., ICCV 2017).

    Output matches nn.BCEWithLogitsLoss(reduction="none") element-wise, so it
    can replace self.bce in v8DetectionLoss without changing normalization.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = pred.sigmoid()
        p_t = p * target + (1 - p) * (1 - target)
        w = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            w = w * a_t
        return w * ce


def apply_focal(detection_model, gamma: float = 2.0, alpha: float = 0.25):
    """Replace the classification BCE with focal loss on both one2many and one2one heads."""
    from ultralytics.utils.loss import E2ELoss

    crit = E2ELoss(detection_model)
    for sub in (crit.one2many, crit.one2one):
        sub.bce = FocalBCE(gamma=gamma, alpha=alpha)
    detection_model.criterion = crit
    print(f"[apply_focal] focal loss applied (gamma={gamma}, alpha={alpha})")
    return crit


# Loading and evaluation notes
# best.pt stores the full DetectionModel (pickle), including the injected
# TridentBlock / BiFlowDetect modules and their weights. Evaluate with
# `YOLO(best.pt)` directly, provided yolo_improved has been imported so the
# custom classes can be deserialized.
