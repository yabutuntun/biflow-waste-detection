# SPDX-License-Identifier: MIT
"""
BiFlow: pairwise bidirectional residual feature compensation.

Given two adjacent feature levels, BiFlow returns residual-compensated
features for both levels. The module depends only on PyTorch.
"""

import torch
import torch.nn as nn

__all__ = ["BiFlow", "BiFlowHead"]


class BiFlow(nn.Module):
    """Pairwise bidirectional residual compensation between two feature levels."""

    def __init__(self, c_high: int, c_low: int):
        super().__init__()
        mid = max((c_high + c_low) // 4, 32)

        # High-to-low branch: stride-2 3x3 convolution downsamples f_high.
        self.high_to_low = nn.Sequential(
            nn.Conv2d(c_high, mid, 3, 2, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c_low, 1, bias=False),
        )

        # Low-to-high branch: bilinear x2 upsample of f_low.
        self.low_to_high = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_low, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c_high, 1, bias=False),
        )

    def forward(self, f_high, f_low):
        return f_high + self.low_to_high(f_low), f_low + self.high_to_low(f_high)


class BiFlowHead(nn.Module):
    """Two BiFlow units applied across P3/P4/P5 in front of the detection head."""

    def __init__(self, c3: int, c4: int, c5: int):
        super().__init__()
        self.biflow1 = BiFlow(c3, c4)
        self.biflow2 = BiFlow(c4, c5)

    def forward(self, features):
        p3, p4, p5 = features
        p3_f, p4_e = self.biflow1(p3, p4)
        p4_f, p5_f = self.biflow2(p4_e, p5)
        return [p3_f, p4_f, p5_f]
