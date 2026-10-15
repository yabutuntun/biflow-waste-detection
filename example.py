"""Minimal usage example for BiFlow."""

import torch

from biflow import BiFlow, BiFlowHead


def main() -> None:
    f_high = torch.randn(1, 64, 80, 80)
    f_low = torch.randn(1, 128, 40, 40)

    pair = BiFlow(c_high=64, c_low=128)
    y_high, y_low = pair(f_high, f_low)
    print("pair:", tuple(y_high.shape), tuple(y_low.shape))

    p3 = torch.randn(1, 64, 80, 80)
    p4 = torch.randn(1, 128, 40, 40)
    p5 = torch.randn(1, 256, 20, 20)
    head = BiFlowHead(c3=64, c4=128, c5=256)
    outputs = head([p3, p4, p5])
    print("head:", [tuple(item.shape) for item in outputs])


if __name__ == "__main__":
    main()
