import torch.nn as nn
import torch.nn.functional as F

""" Building blocks

    ResBlock1D and ResBlock2D are based on the original ResNet paper:

    "Deep residual learning for image recognition"
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
    
    Link to the paper:
    https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
"""

class ResBlock1D(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, stride, activation, dropout):
        super().__init__()

        # First conv
        layers = []
        layers.append(nn.Conv1d(in_chan, out_chan, kernel, stride, padding=1))
        layers.append(nn.BatchNorm1d(out_chan))
        layers.append(activation)
        if dropout > 0.:
            layers.append(nn.Dropout1d(p=dropout))
        self.conv1 = nn.Sequential(*layers)

        # Second conv
        layers = []
        layers.append(nn.Conv1d(out_chan, out_chan, kernel, 1, padding="same"))
        layers.append(nn.BatchNorm1d(out_chan))
        layers.append(activation)
        if dropout > 0.:
            layers.append(nn.Dropout1d(p=dropout))
        self.conv2 = nn.Sequential(*layers)

        # Used for identity mapping
        self.need_downsample = False
        if stride != 1 or in_chan != out_chan:
            self.need_downsample = True

        # No dropout here
        self.downsample = nn.Sequential(
            nn.Conv1d(in_chan, out_chan, kernel, stride, padding=1),
            nn.BatchNorm1d(out_chan)
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        # If downsample, also downsample identity
        if self.need_downsample:
            identity = self.downsample(identity)
        # Skip connection
        x = x + identity
        return x


class ResBlock2D(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, stride, activation, dropout):
        super().__init__()

        # First conv
        layers = []
        layers.append(nn.Conv2d(in_chan, out_chan, kernel, stride, padding=0))
        layers.append(nn.BatchNorm2d(out_chan))
        layers.append(activation)
        if dropout > 0.:
            layers.append(nn.Dropout2d(p=dropout))
        self.conv1 = nn.Sequential(*layers)

        # Second conv
        layers = []
        layers.append(nn.Conv2d(out_chan, out_chan, kernel, 1, padding="same"))
        layers.append(nn.BatchNorm2d(out_chan))
        layers.append(activation)
        if dropout > 0.:
            layers.append(nn.Dropout2d(p=dropout))
        self.conv2 = nn.Sequential(*layers)

        # Used for identity mapping
        self.need_downsample = False
        if stride != 1 or in_chan != out_chan:
            self.need_downsample = True

        # No dropout here
        self.downsample = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel, stride, padding=0),
            nn.BatchNorm2d(out_chan)
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        # If downsample, also downsample identity
        if self.need_downsample:
            identity = self.downsample(identity)
        # Skip connection
        x = x + identity
        return x