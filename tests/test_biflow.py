import torch

from biflow import BiFlow, BiFlowHead


def test_biflow_shapes() -> None:
    f_high = torch.randn(1, 64, 80, 80)
    f_low = torch.randn(1, 128, 40, 40)
    y_high, y_low = BiFlow(c_high=64, c_low=128)(f_high, f_low)
    assert y_high.shape == f_high.shape
    assert y_low.shape == f_low.shape


def test_biflow_head_shapes() -> None:
    p3 = torch.randn(1, 64, 80, 80)
    p4 = torch.randn(1, 128, 40, 40)
    p5 = torch.randn(1, 256, 20, 20)
    outputs = BiFlowHead(c3=64, c4=128, c5=256)([p3, p4, p5])
    assert [tuple(item.shape) for item in outputs] == [
        tuple(p3.shape),
        tuple(p4.shape),
        tuple(p5.shape),
    ]
