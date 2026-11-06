"""BiFlow ablation variants used in the component study.

  baseline       no BiFlow (original Detect head)
  l2h            low-to-high branch only
  h2l            high-to-low branch only
  biflow1        first BiFlow unit only (P3 <-> P4)
  biflow2        second BiFlow unit only (P4 <-> P5)
  full           complete BiFlow (reuses inject_biflow, identical to group D)

The single-direction variants build only the branches that actually take part
in the forward pass, so their parameter and FLOPs counts are real. _build_bu
and _build_td match BiFlow.bu / BiFlow.td layer by layer.
"""

import torch.nn as nn
from ultralytics.nn.modules.head import Detect

from yolo_improved import BiFlow, inject_biflow


def _mid(c_high: int, c_low: int) -> int:
    return max((c_high + c_low) // 4, 32)


def _build_bu(c_high: int, c_low: int) -> nn.Module:
    """Low-to-high upsampling branch, matching BiFlow.bu exactly."""
    mid = _mid(c_high, c_low)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c_low, mid, 1, bias=False),
        nn.BatchNorm2d(mid),
        nn.SiLU(),
        nn.Conv2d(mid, c_high, 1, bias=False),
    )


def _build_td(c_high: int, c_low: int) -> nn.Module:
    """High-to-low downsampling branch, matching BiFlow.td exactly."""
    mid = _mid(c_high, c_low)
    return nn.Sequential(
        nn.Conv2d(c_high, mid, 3, 2, 1, bias=False),
        nn.BatchNorm2d(mid),
        nn.SiLU(),
        nn.Conv2d(mid, c_low, 1, bias=False),
    )


def _detect_channels(detect) -> list:
    return [detect.cv2[i][0].conv.in_channels for i in range(3)]


# Variant 1: low-to-high only
class BiFlowL2HDetect(Detect):
    def forward(self, x):
        p3, p4, p5 = x
        p3_f = p3 + self.l2h1(p4)   # P4 semantics upsampled to P3
        p4_f = p4 + self.l2h2(p5)   # P5 semantics upsampled to P4
        return super().forward([p3_f, p4_f, p5])


def inject_biflow_l2h(detection_model):
    detect = detection_model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(detect).__name__}")
    chs = _detect_channels(detect)
    print(f"[inject_biflow_l2h] P3/P4/P5 = {chs}")
    detect.add_module("l2h1", _build_bu(chs[0], chs[1]))
    detect.add_module("l2h2", _build_bu(chs[1], chs[2]))
    detect.__class__ = BiFlowL2HDetect
    return detect


# Variant 2: high-to-low only
class BiFlowH2LDetect(Detect):
    def forward(self, x):
        p3, p4, p5 = x
        p4_f = p4 + self.h2l1(p3)   # P3 detail downsampled to P4
        p5_f = p5 + self.h2l2(p4)   # P4 detail downsampled to P5
        return super().forward([p3, p4_f, p5_f])


def inject_biflow_h2l(detection_model):
    detect = detection_model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(detect).__name__}")
    chs = _detect_channels(detect)
    print(f"[inject_biflow_h2l] P3/P4/P5 = {chs}")
    detect.add_module("h2l1", _build_td(chs[0], chs[1]))
    detect.add_module("h2l2", _build_td(chs[1], chs[2]))
    detect.__class__ = BiFlowH2LDetect
    return detect


# Variant 3: BiFlow1 only (P3 <-> P4)
class BiFlow1OnlyDetect(Detect):
    def forward(self, x):
        p3, p4, p5 = x
        p3_f, p4_f = self.biflow1(p3, p4)
        return super().forward([p3_f, p4_f, p5])


def inject_biflow1(detection_model):
    detect = detection_model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(detect).__name__}")
    chs = _detect_channels(detect)
    print(f"[inject_biflow1] P3/P4 = {chs[0]}/{chs[1]}")
    detect.add_module("biflow1", BiFlow(chs[0], chs[1]))
    detect.__class__ = BiFlow1OnlyDetect
    return detect


# Variant 4: BiFlow2 only (P4 <-> P5)
class BiFlow2OnlyDetect(Detect):
    def forward(self, x):
        p3, p4, p5 = x
        p4_f, p5_f = self.biflow2(p4, p5)
        return super().forward([p3, p4_f, p5_f])


def inject_biflow2(detection_model):
    detect = detection_model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(detect).__name__}")
    chs = _detect_channels(detect)
    print(f"[inject_biflow2] P4/P5 = {chs[1]}/{chs[2]}")
    detect.add_module("biflow2", BiFlow(chs[1], chs[2]))
    detect.__class__ = BiFlow2OnlyDetect
    return detect


# Variant 5: budget-matched control (in-place refinement, no cross-scale flow).
# Separates "cross-scale bidirectional flow" from "added parameters": adds an
# in-place convolution (residual, no cross-scale connection) to P3/P4/P5 so the
# parameter count matches full BiFlow (+215,616). If full still outperforms it,
# the gain comes from the flow design rather than from added parameters alone.
def _build_refine3x3(c: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c, c, 3, 1, 1, bias=False),
        nn.BatchNorm2d(c),
        nn.SiLU(),
    )


def _build_refine1x1(c: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c, c, 1, bias=False),
        nn.BatchNorm2d(c),
        nn.SiLU(),
    )


class BiFlowBudgetMatchedDetect(Detect):
    def forward(self, x):
        p3, p4, p5 = x
        p3_f = p3 + self.bm3(p3)
        p4_f = p4 + self.bm4(p4)
        p5_f = p5 + self.bm5(p5)
        return super().forward([p3_f, p4_f, p5_f])


def inject_biflow_budget_matched(detection_model):
    detect = detection_model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError(f"Last layer is not Detect: {type(detect).__name__}")
    chs = _detect_channels(detect)
    print(f"[inject_biflow_budget_matched] P3/P4/P5 = {chs}")
    detect.add_module("bm3", _build_refine3x3(chs[0]))
    detect.add_module("bm4", _build_refine3x3(chs[1]))
    detect.add_module("bm5", _build_refine1x1(chs[2]))
    detect.__class__ = BiFlowBudgetMatchedDetect
    return detect


# Unified entry point
VARIANTS = {
    "baseline": None,          # no injection
    "l2h": inject_biflow_l2h,
    "h2l": inject_biflow_h2l,
    "biflow1": inject_biflow1,
    "biflow2": inject_biflow2,
    "budget_matched": inject_biflow_budget_matched,
    "full": inject_biflow,     # reuse the complete BiFlow
}


def inject_ablation(detection_model, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; options: {sorted(VARIANTS)}")
    fn = VARIANTS[variant]
    return fn(detection_model) if fn is not None else None
