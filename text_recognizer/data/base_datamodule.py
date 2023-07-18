from pathlib import Path
from typing import Collection, Union, Tuple, Dict, Optional
import argparse

from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl

from text_recognizer import utils
from text_recognizer.data.util import BaseDataset

BATCH_SIZE = 128
NUM_WORKERS = 0

def load_and_print_info(data_module_class) -> None:
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # set the below varibles in the subclass
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "raw_data"
        
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="batch size of dataset"
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="num of additional process to add data"
        )
        return parser
        
    def config(self):
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}
    
    def prepare_data(self, *args, **kwargs) -> None:
        """
        to prepare the data
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        setup the data
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )
    
    def val_datalaoder(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )
        
    def test_datalaoder(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )