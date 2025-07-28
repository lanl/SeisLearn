    @staticmethod
    def picks_from_annotations(annotations, threshold, phase) -> util.PickList:
        """
        Converts the annotations streams for a single phase to discrete picks using a classical trigger on/off.
        The lower threshold is set to half the higher threshold.
        Picks are represented by :py:class:`~seisbench.util.annotations.Pick` objects.
        The pick start_time and end_time are set to the trigger on and off times.

        :param annotations: Stream of annotations
        :param threshold: Higher threshold for trigger
        :param phase: Phase to label, only relevant for output phase labelling
        :return: List of picks
        """
        picks = []
        for trace in annotations:
            trace_id = (
                f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
            )
            triggers = trigger_onset(trace.data, threshold, threshold / 2)
            times = trace.times()
            for s0, s1 in triggers:
                t0 = trace.stats.starttime + times[s0]
                t1 = trace.stats.starttime + times[s1]

                peak_value = np.max(trace.data[s0 : s1 + 1])
                s_peak = s0 + np.argmax(trace.data[s0 : s1 + 1])
                t_peak = trace.stats.starttime + times[s_peak]

                pick = util.Pick(
                    trace_id=trace_id,
                    start_time=t0,
                    end_time=t1,
                    peak_time=t_peak,
                    peak_value=peak_value,
                    phase=phase,
                )
                picks.append(pick)

        return util.PickList(sorted(picks))



import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset


class ShiftDataset(Dataset):
    def __init__(self, x, y, win_length):
        self.x = x
        self.y = y
        self.win_length = win_length
        self.low = 0
        self.high = x.shape[-1] - win_length

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # This does the shuffling
        istart = torch.randint(low=self.low, high=self.high, size=(1,)).item()
        iend = istart + self.win_length
        return (self.x[idx, :, istart:iend], self.y[idx, istart:iend])

# Your data should be stored as something like this
path = None
win_length = 3000
x = np.load(path) # dim = (n_samples, n_channels, waveform_length), e.g., (1M, 3, 6000)

# Make it into a pytorch tensors (you can also just store the data as .pt and read directly)
x = torch.from_numpy(x).float()

n_samples, n_channels, waveform_length = x.shape

# Now, you make y
y = torch.zeros((n_samples, 1, waveform_length), dtype=torch.float32)
# Put your Gaussian kernel to y

# Now, make the dataset object
dataset = ShiftDataset(x, y, win_length)

# Now, make the data loader
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# In train/val loop, do
for x, y in data_loader:
    ...
