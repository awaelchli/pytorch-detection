import torch
from layer import Detection


def test_heatmap():
    layer = Detection(1, 1)
    tensor = torch.zeros(3, 1, 5, 5)
    tensor[0, 0, 1, 1] = 1.111
    tensor[1, 0, 2, 2] = 2.222
    tensor[2, 0, 4, 0] = 3.333

    top, loc = layer(tensor)

    assert top[0, 0, 0] == 1.111
    assert top[1, 0, 0] == 2.222
    assert top[2, 0, 0] == 3.333

    assert (loc[0, 0, 0], loc[0, 1, 0]) == (-0.5, 0.5)
    assert (loc[1, 0, 0], loc[1, 1, 0]) == (0.0, 0.0)
    assert (loc[2, 0, 0], loc[2, 1, 0]) == (-1.0, -1.0)


def test_channels():
    layer = Detection(3, 1)
    tensor = torch.zeros(1, 3, 3, 3)

    # heatmap
    tensor[0, 0, 1, 0] = 400
    tensor[0, 0, 1, 1] = 300
    tensor[0, 0, 1, 2] = 600

    # feature channels
    tensor[0, 1] = torch.rand(3, 3)
    tensor[0, 2] = torch.rand(3, 3)

    top, loc = layer(tensor)

    assert top[0, 0, 0] == 600
    assert top[0, 1, 0] == tensor[0, 1, 1, 2]
    assert top[0, 2, 0] == tensor[0, 2, 1, 2]


if __name__ == '__main__':
    test_heatmap()
    test_channels()
