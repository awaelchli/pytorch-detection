import torch
import torch.nn as nn


class Detection(nn.Module):

    def __init__(self, in_channels, k):
        super(Detection, self).__init__()
        self.in_channels = in_channels
        self.k = k

    def grid(self, height, width):
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(1, -1, height)

        x = x.view(1, -1).repeat(height, 1)
        y = y.view(-1, 1).repeat(1, width)
        return x, y

    def forward(self, input):
        # input shape: [batch_size, in_channels, height, width]
        assert input.size(1) == self.in_channels
        batch_size = input.size(0)
        height = input.size(2)
        width = input.size(3)

        input_flat = input.view(batch_size, self.in_channels, -1)
        heatmap = input_flat[:, 0]
        _, indices = torch.topk(heatmap, self.k, dim=1)

        feature_indices = indices.unsqueeze(1).expand(-1, self.in_channels, -1)
        features = input_flat.gather(2, feature_indices)

        x, y = self.grid(height, width)
        x_flat = x.view(1, -1)
        y_flat = y.view(1, -1)
        x_flat = x_flat.expand(batch_size, -1)
        y_flat = y_flat.expand(batch_size, -1)

        x_location = x_flat.gather(1, indices)
        y_location = y_flat.gather(1, indices)
        location = torch.stack((x_location, y_location), dim=2)
        location = location.transpose(1, 2)

        # features shape: [batch_size, in_channels, k]
        # location shape: [batch_size, 2, k]
        return features, location


if __name__ == '__main__':
    layer = Detection(3, 4)

    input = torch.rand(2, 3, 10, 15)
    output, location = layer(input)

    print(output.size())
    print(location.size())
    print(output)
