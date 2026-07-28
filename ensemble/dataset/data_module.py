from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_datasets


class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        train_dataset,val_dataset,test_dataset,predict_dataset = create_datasets(self.args)
        self.dataset = {
            "train":train_dataset,
            "validation":val_dataset,
            "test":test_dataset,
            "pred":predict_dataset,
        }

    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,shuffle = True, 
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader

    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader

    def predict_dataloader(self):
        loader = DataLoader(self.dataset["pred"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader