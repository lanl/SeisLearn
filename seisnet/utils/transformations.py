from torchaudio.transforms import Spectrogram
import torch

def spectrogram_transform(tensor: torch.Tensor, 
                          n_fft:int = 250, 
                          win_length:int = 250, 
                          hop_length:int = 4)->torch.Tensor:
    """
    Create spectrogram from torch tensor. Tensor input shape should be 
    (n_samples, in_chan, win_len)
    n_samples - number of rows in training data
    in_chan - number of waveform channels e.g. 3 for 3-components
    win_len - window length of samples e.g 3000 for 30s of data sampled at 100Hz
    """
    # These are the exact parameters used for converting 500 samples long window
    # into 126 x 126 spectrogram
    transform = Spectrogram(n_fft=n_fft, win_length=win_length, 
                            hop_length=hop_length).float()
    xspec = transform(tensor)
    return xspec