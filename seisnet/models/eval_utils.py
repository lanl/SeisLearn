from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from detecta import detect_peaks
from torchmetrics import Accuracy, F1Score, FBetaScore, Recall
from tqdm import tqdm

from seisnet.dataloaders import test_dataloader


def eval_loss_workflow(model:nn.Module,data_dir_regex:str,loss_fn,
                       classes:int,batch_size:int=256,samples_thresh:int=10,
                       prob_thresh:float=0.5,named:str="all",dp:int=8):
    """
    Evaluate a trained model on the test dataset

    Params:
        model(nn.Module): Pretained phasenet model
        data_dir_regex(str): regex path for test dataset waveforms. 
                            Can include wildcards like `*`
        loss_fn(object): Class or function for evaluating loss
        classes(int): Number of classes that should be predicted. Typically 1 or 3
        batch_size(int): Batch size
        samples_thresh(int): Number of samples around phase arrivals to 
                             create binary prediction classification
        prob_thresh(float): Minimum prediction thresh to detect peaks
        named(str): Name of test dataset
    
    Returns:
        (float): test_loss value
    """
    dataloader = test_dataloader(data_dir_regex,batch_size,classes)
    num_batches = len(dataloader)
    test_loss = 0
    
    # Partial peaks function
    peaks_thresh = partial(detect_peaks, mph=prob_thresh, mpd=500, show=False)

    pred_class = [] 
    obs_class = []
    residuals = []

    # Prevent weight updates
    model.eval()
    

    with torch.no_grad():
        for batch in tqdm(dataloader,position=0,desc=f"performing inference on {named} waveforms ...."):
            pred = model(batch["X"].to(model.device))
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()
            plabels = batch["pIdx"].detach().numpy()

            for idx,arr in enumerate(pred.squeeze(1)):
                onsets = peaks_thresh(arr)
                # Only one peak is detected
                if len(onsets)==1:
                    err = onsets[0]-plabels[idx]
                    if abs(err) <= samples_thresh:
                        pred_class.append(1)
                        obs_class.append(1)
                    else:
                        pred_class.append(0)
                        obs_class.append(1)
                    residuals.append(err)
                    continue
                # More than one peak is detected
                elif len(onsets)>1:
                    errs = np.array(onsets)-plabels[idx]
                    errs_abs = np.abs(errs)
                    closest = errs[np.argmin(errs_abs)]
                    residuals.append(closest)
                    if abs(closest) <= samples_thresh:
                        pred_class.append(1)
                        obs_class.append(1)
                    else:
                        pred_class.append(0)
                        obs_class.append(1)
                    pred_class.extend(np.ones(len(errs)-1))
                    obs_class.extend(np.zeros(len(errs)-1))
                    continue
                # No peak is detected
                else:
                    pred_class.append(0)
                    obs_class.append(1)


    # Compute the metrics
    test_loss /= num_batches
    f1 = F1Score(task="binary")
    fbeta = FBetaScore(task="binary")
    recall = Recall(task="binary")
    accuracy = Accuracy(task="binary")

    target_tensor = torch.tensor(obs_class, device=model.device).to(torch.int)
    pred_tensor = torch.tensor(pred_class, device=model.device).to(torch.float)

    fl_score = f1(pred_tensor, target_tensor).item()
    fbeta_score = fbeta(pred_tensor, target_tensor).item()
    rec_score = recall(pred_tensor, target_tensor).item()
    acc_score = accuracy(pred_tensor, target_tensor).item()

    # Save the parameters as a dict
    params = {
            f"{named}_test_loss": round(test_loss,dp),
            f"{named}_recall": round(rec_score,dp),
            f"{named}_f1": round(fl_score,dp),
            f"{named}_fbeta": round(fbeta_score,dp),
            f"{named}_acc": round(acc_score,dp),
    }

    df = pd.DataFrame({"residuals":residuals})
    df.to_csv(f"{named}_residuals.csv", index=False)
    
    return params
