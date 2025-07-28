import torch.nn as nn
import torch.nn.functional as F

from seisnet.models import ResBlock1D, ResBlock2D


def make_1d_resblock_layers(win_len, in_chan, mid_chans, kernel, stride, activation, dropout):
    """
    A wrapper for making 1D ResNet encoders
    """
    layers = []
    curr_chan = in_chan
    curr_len = win_len
    for next_chan in mid_chans:
        layers.append(ResBlock1D(
            curr_chan, next_chan, kernel, stride,
            activation=activation, dropout=dropout
        ))
        curr_chan = next_chan
        curr_len = calc_Lout(curr_len, kernel, stride, 1, 1)
    flat_len = curr_len * mid_chans[-1]
    return layers, flat_len


def make_2d_resblock_layers(height, width, in_chan, mid_chans, kernel, stride, activation, dropout):
    """
    A wrapper for making 2D ResNet encoders
    """
    layers = []
    curr_chan = in_chan
    curr_height = height
    curr_width = width
    for next_chan in mid_chans:
        layers.append(ResBlock2D(
            curr_chan, next_chan, kernel, stride,
            activation=activation, dropout=dropout
        ))
        curr_chan = next_chan
        curr_height, curr_width = calc_HWout(curr_height, curr_width, kernel, stride, 0, 1)
    flat_len = curr_height * curr_width * mid_chans[-1]
    return layers, flat_len


def make_linear_layers(flat_len, hdims, out_len, activation):
    """
    A wrapper for making linear layers
    """
    layers = []
    layers.append(nn.Flatten())
    curr_dim = flat_len
    for hdim in hdims:
        layers.append(nn.Linear(curr_dim, hdim))
        layers.append(nn.BatchNorm1d(hdim))
        layers.append(activation)
        curr_dim = hdim
    layers.append(nn.Linear(curr_dim, out_len))
    return layers


def calc_Lout(Lin, kernel, stride, padding, dilation):
    """ 
    Calculate the length of the tensor output after Conv1D
    """
    return (Lin + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def calc_HWout(Hin, Win, kernel, stride, padding, dilation):
    """ 
    Calculate the height and width of the tensor output after Conv2D
    """
    if isinstance(kernel, int):
        kernel = [kernel, kernel]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]
    if isinstance(dilation, int):
        dilation = [dilation, dilation]
    Hout = (Hin + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    Wout = (Win + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    return Hout, Wout