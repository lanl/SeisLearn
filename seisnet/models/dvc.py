import torch.nn as nn
import torch.nn.functional as F

from seisnet.utils import (make_1d_resblock_layers, 
                           make_2d_resblock_layers,
                           make_linear_layers)

""" Supplemental material for
    
    "Divide and conquer: Separating the two probabilities in seismic phase picking"
        by Yongsoo Park, Alysha Armstrong, William Yeck, David Shelly, and Gregory Beroza
    
    Link to the paper: TBD
    File created by Yongsoo Park (ysp@lanl.gov)

    NOTE:
    * This is a code snippet of the neural networks used in the divide and conquer approach
    
    * The ProbPhaseExist class is the binary classifier used for
        addressing the probability of existence of a phase arrival
    
    * The ProbPhaseArrival class is the multi-class classifier used for
        addressing the probability of an arrival time (outputs a probability mass function)
"""

class ProbPhaseExist(nn.Module):
    """ A neural network for addressing the probability of existence of a phase arrival
        
        NOTE:
        * The parameters used in the study are set as default values
        * The input is a spectrogram from:
            from torchaudio.transforms import Spectrogram
            transform = Spectrogram(n_fft=250, win_length=250, hop_length=4).float()
    """
    def __init__(
            self,
            height=126,
            width=126,
            in_chan=3,
            mid_chans=[6, 12, 24],
            kernel=7,
            stride=2,
            hdims=[64, 32],
            activation=nn.ReLU(),
            dropout=0.05,
            apply_sigmoid=True
        ):
        
        super().__init__()

        # Resnet layers
        layers, flat_len = make_2d_resblock_layers(
            height, width, in_chan, mid_chans, kernel, stride, activation, dropout
        )
        self.res_blocks = nn.Sequential(*layers)

        # Linear layers
        self.linear_layers = nn.Sequential(*make_linear_layers(flat_len, hdims, 1, activation))

        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, x):
        x = self.linear_layers(self.res_blocks(x))
        if self.apply_sigmoid:
            x = F.sigmoid(x).squeeze(-1)
        else:
            x = x.squeeze(-1)
        return x


class ProbPhaseArrival(nn.Module):
    """ A neural network for addressing the probability of an arrival time

        NOTE:
        * The parameters used in the study are set as default values
        * The SoftMax is applied to the window dimension, giving a probability mass function
    """
    def __init__(
            self,
            win_len=500,
            in_chan=3,
            mid_chans=[6, 12, 24],
            kernel=7,
            stride=2,
            hdims=[64, 32],
            activation=nn.ReLU(),
            dropout=0.05,
            apply_softmax=True
        ):
        super().__init__()

        # Resnet layers
        layers, flat_len = make_1d_resblock_layers(
            win_len, in_chan, mid_chans, kernel, stride, activation, dropout
        )
        self.res_blocks = nn.Sequential(*layers)

        # Linear layers
        self.linear_layers = nn.Sequential(*make_linear_layers(flat_len, hdims, win_len, activation))

        self.apply_softmax = apply_softmax
    
    def forward(self, x):
        x = self.linear_layers(self.res_blocks(x))
        if self.apply_softmax:
            x = F.softmax(x, dim=-1)
        return x