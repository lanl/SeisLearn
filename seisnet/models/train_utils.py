import os
import time

import mlflow
from mlflow.models import infer_signature
import torch
from loguru import logger
from tqdm import tqdm
import torch.distributed as dist
from glob import glob

from seisnet.utils import get_repo_dir

def reduce_tensor(tensor):
    rt = tensor.clone()
    # Sum loss values from all processes
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    rt /= world_size
    return rt

def train_loop(
        model, dataloader, loss_fn, 
        optimizer, rank, epoch:int, 
        total_epochs:int, verbose:bool=True):
    """
    Training loop to perform back propagation and update weights
    """
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    running_loss = 0
    num_batches = len(dataloader)
    for batch_idx, (batch) in enumerate(progress_bar):
        if torch.cuda.is_available():
            pred = model(batch["X"].to(rank))
            loss = loss_fn(pred, batch["y"].to(rank))
        else:
            pred = model(batch["X"].to(model.device))
            loss = loss_fn(pred, batch["y"].to(model.device))
        

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ✅ Average loss across GPUs
        with torch.no_grad():
            reduced_loss = reduce_tensor(loss.detach())

        running_loss += reduced_loss.item()
        avg_loss = running_loss / (batch_idx+1)

        if rank == 0:
            progress_bar.set_postfix({"loss": f"{avg_loss:>.8f}"})
    
    epoch_loss = running_loss / num_batches

    if rank == 0:
        mlflow.log_metric("loss",round(epoch_loss,5),step=epoch+1)
        if verbose:
            print(f"Epoch {epoch+1}/{total_epochs} - train loss: {epoch_loss:>.8f}")
    
    return avg_loss


def test_loop(model, dataloader, loss_fn, rank, epoch:int, verbose:bool=True):
    """
    Evaluation loop for validation/test inference
    """
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            if torch.cuda.is_available():
                pred = model(batch["X"].to(rank))
                tmp_loss = loss_fn(pred, batch["y"].to(rank))
                reduced_loss = reduce_tensor(tmp_loss.detach())
                test_loss += reduced_loss.item()
            else:
                pred = model(batch["X"].to(model.device))
                test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage

    test_loss /= num_batches

    if rank==0:
        mlflow.log_metric("val_loss",round(test_loss,5),step=epoch+1)
        if verbose:
            print(f"Validation avg loss: {test_loss:>.8f} \n-------------------------------")
    
    return test_loss


def vector_cross_entropy(y_pred, y_true, eps=1e-5):
    """
    vector cross entropy loss
    """
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()            # Mean over batch axis
    return -h


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least loss, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float("inf"), verbose=True):
        self.best_valid_loss = best_valid_loss
        self.verbose = verbose
        self.patience = 0

        save_dir = f"{get_repo_dir()}/outputs"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        
    def __call__(
        self, current_valid_loss, identifier,
        epoch, model, optimizer, data_size, signature=None
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            if self.verbose:
                print(f"Best validation loss: {self.best_valid_loss:>.8f}")
                print(f"Saving best model for epoch: {epoch+1}\n")
            # Any type of device gpu/cpu
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "data_size": data_size
                }, 
                f"{self.save_dir}/best_model_s{data_size:d}_{identifier}.pth")
            
            # Logging the model with mlflow is slower and creates an annoying output
            # mlflow.pytorch.log_model(model, "model", signature=signature, 
            #                          registered_model_name=f"best_model_{identifier}")
            mlflow.log_metric("best_epoch",epoch+1)
            self.patience = 0
        else: 
            self.patience += 1


class AdaptiveEarlyStopping:
    """
    This setup prevents abrupt stopping when performance is likely to 
    improve with a few more epochs, especially useful for highly 
    fluctuating metrics
    """
    def __init__(self, base_patience=10, delta=0.01, verbose=True):
        self.base_patience = base_patience
        self.delta = delta
        self.verbose = verbose
        self.wait_count = 0
        self.best_score = None
        self.dynamic_patience = self.base_patience
    
    def step(self, val_loss):
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.wait_count = 0
            self.dynamic_patience = self.base_patience  # reset to base
        else:
            self.wait_count += 1
            # Adjust patience if improvement is near
            if self.wait_count >= (self.base_patience * 0.8):
                self.dynamic_patience += 1
            if self.wait_count >= self.dynamic_patience:
                if self.verbose:
                    print("Stopping early due to lack of improvement.")
                return True  # Signal to stop training
        return False


def train_model_cycle(model_uuid, model, loss_fn, optimizer, scheduler, 
                      rank, train_dataloader, validation_dataloader, 
                      max_epochs, data_size, verbose:bool=True)->dict:
    """
    Training cycle that saves the training history and best models
    verbose(bool) - Print out results
    """
    
    # initialize custom classes
    save_best_model = SaveBestModel(verbose=verbose)
    early_stopping = AdaptiveEarlyStopping(base_patience=10, delta=0.01,verbose=verbose)
    #  Get the model signature for mlflow
    # batch = next(iter(train_dataloader))
    # signature = infer_signature(batch["X"].detach().numpy(), batch["y"].detach().numpy())
    signature= None

    epochs_hist = []
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(max_epochs):
        # Train the model
        train_loss = train_loop(model, train_dataloader, loss_fn, optimizer, 
                                rank=rank, epoch=epoch, total_epochs=max_epochs, 
                                verbose=verbose)
        
        # Validate the model
        val_loss = test_loop(model, validation_dataloader, loss_fn, 
                             rank=rank, epoch=epoch, verbose=verbose)

        lr_tensor = torch.tensor([0.0], dtype=torch.float32).to(rank)

        if rank==0:
            # Save the best model
            save_best_model(
                val_loss, model_uuid, epoch,  model, 
                optimizer=optimizer, 
                data_size=data_size, 
                signature=signature
            )

            # Update the learning rate scheduler AFTER saving model
            scheduler.step(val_loss)
            lr_tensor.fill_(optimizer.param_groups[0]['lr'])
            mlflow.log_metric("lr", lr_tensor.item(), step=epoch+1)

            # Check Early Stopping
            dyn_stop = early_stopping.step(val_loss)
            
            epochs_hist.append(epoch+1)
            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            # Save checkpoints every 
            if epoch>0 and (epoch+1)%100==0:
                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "data_size": data_size
                    }, 
                    f"{save_best_model.save_dir}/model_s{data_size:d}_chkpt{epoch+1}_{model_uuid}.pth")
            
            # Prevent hardcoded early stopping
            if dyn_stop:# or save_best_model.patience >= early_stopping.base_patience
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # # All ranks update optimizer LR
        dist.broadcast(lr_tensor, src=0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_tensor.item()
    
    history = { 
        "epochs": epochs_hist,
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "data_size": data_size
    }

    time.sleep(2)
    if rank==0:
        logger.info(f"Experiment {model_uuid} completed. Exporting history ...")
        check_core_files = glob(f"{get_repo_dir()}/core.*")
        if len(check_core_files): # Delete files that can drain memory 
            for core_file in check_core_files:
                os.remove(core_file)
        
    return history