import torch
import random
from scipy.signal import detrend as scipy_detrend


class Normalize(object):
    """
    Normalize a tensor using either min-max or mean-std across a specific axis.
    Also performs demeaning and detrending.
    """
    def __init__(self, mode: str = "mnstd", axis: int = 1, eps: float = 1e-8):
        assert mode in ["mnmx", "mnstd"], "Invalid mode argument"
        self.mode = mode
        self.axis = axis
        self.eps = eps

    def _demean(self, x):
        mean = x.mean(dim=self.axis, keepdim=True)
        return x - mean

    def _detrend(self, x):
        # Use SciPy detrend on CPU tensor, then convert back to torch
        x_cpu = x.detach().cpu().numpy()
        x_detrended = scipy_detrend(x_cpu, axis=self.axis)
        return torch.tensor(x_detrended, dtype=torch.float32, device=x.device)

    def _normalize(self, x):
        if self.mode == "mnmx":
            min_val, _ = x.min(dim=self.axis, keepdim=True)
            max_val, _ = x.max(dim=self.axis, keepdim=True)
            norm = (x - min_val) / ((max_val - min_val) + self.eps)
        elif self.mode == "mnstd":
            std = x.std(dim=self.axis, keepdim=True)
            norm = x / (std + self.eps)
        return norm

    def __call__(self, sample):
        x, y, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]
        x = self._demean(x)
        x = self._detrend(x)
        x = self._normalize(x)
        return { "X": x, "y": y, "pIdx": pIdx, "sIdx": sIdx }


class RandomShiftChannels(object):
    """
    Randomly shift window to simulate varying arrival times.
    Assumes input shape (channels, time).
    """
    def __init__(self, min_shift: int = 0, max_shift: int = 2800, window: int = 3000):
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.window = window

    def __call__(self, sample):
        x, y, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]
        start = random.randint(self.min_shift, self.max_shift)
        end = start + self.window
        x = x[:, start:end]
        y = y[:, start:end]
        pIdx = self.window - start
        return { "X": x, "y": y, "pIdx": pIdx, "sIdx": sIdx }


class RandomZeroChannels(object):
    """
    Randomly zero out 0–2 channels to simulate sensor dropout or noise.
    """
    def __call__(self, sample):
        x, y, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]
        zeroed_chns = random.randint(0, 2)
        if zeroed_chns > 0:
            x[:zeroed_chns, :] = 0
        return { "X": x, "y": y, "pIdx": pIdx, "sIdx": sIdx }


class ToTensor(object):
    """
    Identity transform when working with preloaded tensors.
    Kept for compatibility with previous pipelines.
    """

    def __init__(self, classes=1):
        self.classes = classes

    def __call__(self, sample):
        x, y, pIdx, sIdx = sample["X"], sample["y"], sample["pIdx"], sample["sIdx"]

        # In legacy mode, select first channel of y
        if self.classes == 1 and y.shape[0] == 3:
            y = y[0:1]

        # Ensure tensor types
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        return { "X": x, "y": y, "pIdx": pIdx, "sIdx": sIdx }
