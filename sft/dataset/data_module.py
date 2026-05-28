from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from dataset.data_helper import create_datasets
from config.config import parser
import torch
import numpy as np
from torch.utils.data.sampler import RandomSampler
import math
import random


class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        train, dev, test = create_datasets(self.args)
        self.dataset = {
            "train": train, "validation": dev, "test": test
        }

    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """

        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=False,shuffle=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader

    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """

        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=True, pin_memory=False,
                            shuffle=False,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader



