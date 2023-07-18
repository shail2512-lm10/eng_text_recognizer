from pathlib import Path
from typing import Sequence
import json
import os
import shutil

from torchvision import transforms
import h5py
import numpy as np
import toml

from text_recognizer.data.base_datamodule import BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, split_dataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"

