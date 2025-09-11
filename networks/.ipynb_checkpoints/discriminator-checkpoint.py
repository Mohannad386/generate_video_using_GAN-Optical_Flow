from networks.blocks import DownBlock2d
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512, sn = False):
        super(Discriminator, self).__init__()
        
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1)))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        feature_maps = []
        out = x

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)
        prediction_map = nn.ReLU()(prediction_map)
        prediction_map = nn.Flatten()(prediction_map)
        prediction_map = nn.Linear(prediction_map.shape[-1], 1).to(prediction_map.device)(prediction_map)
        prediction_map = torch.sigmoid(prediction_map)

        return prediction_map