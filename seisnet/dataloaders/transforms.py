import torch
import numpy as np
from scipy.signal import detrend


class Normalize(object):
    """
    Normalize an array either using min-max or mean-std
    valid mode areguments are `mnmx` and `mnstd`
    eps is to prevent zero division in the zscore normalization
    """
    def __init__(self, mode:str="mnstd", axis:int=1, eps:float=1e-8):
        self.mode = mode
        self.eps = eps
        self.axis = axis
        assert self.mode in ["mnmx","mnstd"], "Invalid mode argument"
    
    def _demean(self, x):
        x = x - np.mean(x, axis=self.axis, keepdims=True)
        return x

    def _detrend(self, x):
        x = detrend(x, axis=self.axis)
        return x
    
    def _normalize(self, x):
        if self.mode=="mnmx":
            row_min = x.min(axis=self.axis, keepdims=True)
            row_max = x.max(axis=self.axis, keepdims=True)
            norm = (x-row_min) / ((row_max-row_min) + self.eps)
        elif self.mode=="mnstd":
            # Demean applied above
            # row_mean = x.mean(axis=self.axis, keepdims=True) 
            row_std = x.std(axis=self.axis, keepdims=True)
            norm = x / (row_std + self.eps)
        return norm

    def __call__(self, sample):
        waveform, labels, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]

        waveform = self._demean(waveform)
        waveform = self._detrend(waveform)
        waveform = self._normalize(waveform)

        return { "X": waveform, "y": labels, "pIdx": pIdx, "sIdx": sIdx }
    
class RandomShiftChannels(object):
    """
    Randomly shift channels so the arrival time is not fixed
    """
    def __call__(self, sample):
        waveform, labels, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]
        start = np.random.randint(0,2800)
        end = start+3000
        waveform = waveform[:,start:end]
        labels = labels[:,start:end]
        pIdx= 3000 - start

        return { "X": waveform, "y": labels, "pIdx": pIdx, "sIdx": sIdx }


class RandomZeroChannels(object):
    """
    Randomly zero out 0-2 channels in the dataset
    """
    def __call__(self, sample):
        waveform, labels, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]
        zeroed_chns = np.random.randint(0,2)
        waveform[:zeroed_chns, :] = waveform[:zeroed_chns, :] * 0

        return { "X": waveform, "y": labels, "pIdx": pIdx, "sIdx": sIdx }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,classes=1):
        self.classes = classes

    def __call__(self, sample):
        waveform, labels, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]
        
        if self.classes==1 and labels.shape[0]==3:
            return { 
                "X": torch.from_numpy(waveform).to(torch.float32),
                "y": torch.from_numpy(labels[0]).to(torch.float32).unsqueeze(0), 
                "pIdx": pIdx, "sIdx": sIdx
            }
        else:
            return { 
                "X": torch.from_numpy(waveform).to(torch.float32),
                "y": torch.from_numpy(labels).to(torch.float32), 
                "pIdx": pIdx, "sIdx": sIdx
            }