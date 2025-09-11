from networks.blocks import SameBlock2d, DownBlock2dG, UpBlock2dG, ResBlock2d
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, num_channels_image, num_channels_motion, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks):
        super(Generator, self).__init__()

        self.first1 = SameBlock2d(num_channels_image, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2dG(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks1 = nn.ModuleList(down_blocks)
        
        self.first2 = SameBlock2d(num_channels_motion, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2dG(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks2 = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i))) * 2
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1))) * 2
            up_blocks.append(UpBlock2dG(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks)) * 2
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion * 2, 3, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, source_image, motion_field):
        # Encoding (downsampling) part
        out_source = self.first1(source_image)
        out_motion_field = self.first2(motion_field)
        
        for i in range(len(self.down_blocks1)):
            out_source = self.down_blocks1[i](out_source)
            
        for i in range(len(self.down_blocks2)):
            out_motion_field = self.down_blocks2[i](out_motion_field)
        
        # Concatenating
        out = torch.cat((out_source, out_motion_field), dim = 1)
        
        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = torch.sigmoid(out)

        return out