from pathlib import Path
from typing import Collection, Union, Tuple, Dict, Optional
import argparse

from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl

from text_recognizer import utils
from text_recognizer.data.util import BaseDataset

BATCH_SIZE = 128
NUM_WORKERS = 0

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        