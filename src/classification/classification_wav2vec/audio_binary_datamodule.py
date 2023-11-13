import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from audio_binary_dataset import AudioBinaryDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        target_sample_rate,
        num_samples,
        real_samples: pd.Series,
        cloned_samples: pd.Series,
        max_imbalance=1
    ):
        super().__init__()
        self.batch_size = batch_size
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.cloned_samples = cloned_samples
        self.real_samples = real_samples
        self.max_imbalance = max_imbalance
    
    def prepare_data(self):
        dataset = AudioBinaryDataset(
            self.real_samples,
            self.cloned_samples,
            self.target_sample_rate,
            self.num_samples,
            max_imbalance=self.max_imbalance
        )
        
        self.dataset_training, self.dataset_validation, self.dataset_test = random_split(
            dataset,
            [0.7, 0.1, 0.2],
            generator=torch.Generator().manual_seed(0)
        )
        
    def train_dataloader(self):
        return DataLoader(self.dataset_training, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
