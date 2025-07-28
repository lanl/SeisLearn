from glob import glob

from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from seisnet.dataloaders.datasets import SeisDataset
from seisnet.dataloaders.transforms import Normalize, ToTensor


def test_dataloader(regex_path:str, batch_size:int=64, classes:int=3)->DataLoader:
    """
    Create a test dataloader given an input directory. It uses all the files in the directory

    Arguments:
        wvfm_dir(str):      Placeholder path to npz files that have been cropped to model input length
        batch_size(int):    Batch size for dataloader
        classes(int):       Number of classes that should be predicted. Valid values are 1 or 3. 
                            Depends on how the model was trained
    """
    transforms = Compose([Normalize("mnstd"), ToTensor(classes=classes)])

    all_test_files = natsorted(glob(f"{regex_path}/*.npz"))
    test_dataset = SeisDataset(all_test_files, transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_dataloader