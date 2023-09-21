import argparse
from pathlib import Path
from typing import Optional, Sequence
import json
import os
import shutil
import zipfile

from torchvision import transforms
import h5py
import numpy as np

from text_recognizer.data.base_datamodule import BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, split_dataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"


class EMNIST(BaseDataModule):
    def __init__(self, args = None) -> None:
        super().__init__(args)

        if not os.path.exists(ESSENTIALS_FILENAME):
            _process_emnist(filename= "matlab.zip", dirname=RAW_DATA_DIRNAME)
        
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"])
        self.output_dims = (1,)

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _process_emnist(filename= "matlab.zip", dirname=RAW_DATA_DIRNAME)
        with open(ESSENTIALS_FILENAME) as f:
            _essentials = json.load(f)
    
    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)

            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)
 

def _process_emnist(filename: str, dirname: Path):
    print("Unzipping EMNIST....")
    curdir = os.getcwd()
    os.chdir(dirname)
    zipped = zipfile.ZipFile(filename, "r")
    zipped.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat

    print("loading data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
    x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS

    print("Saving to HDF5 in compressed format")
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    print("saving essential dataset parameters...")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
    characters = _augment_emnist_characters(list(mapping.values))
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}
    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f)

    print("cleaning up...")
    shutil.rmtree("matlab")
    os.chdir(curdir)

def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]
    # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!
    return ["<B>", "<S>", "<E>", "<P>", *characters, *iam_characters]