from abc import ABC

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from constants import TARGET_COL, NUM_WORKERS
from napi_utils import load_data, download_current_data


class NumeraiDataset(Dataset):
    def __init__(self, name="train", feature_set="small", sample_4th_era=True, aux_target_cols=None):
        super().__init__()
        if aux_target_cols is None:
            aux_target_cols = []
        self.df = load_data(name,
                            feature_set=feature_set,
                            sample_4th_era=sample_4th_era,
                            aux_target_cols=aux_target_cols)

        # ensure auxiliary targets have values
        for target_aux_col in aux_target_cols:
            self.df.loc[:, target_aux_col].fillna(0.5, inplace=True)

        # extract feature col names
        feature_cols = [c for c in self.df if c.startswith("feature_")]

        self.inputs = torch.tensor(self.df[feature_cols].values)
        self.targets = torch.tensor(self.df[TARGET_COL].values)
        self.aux_targets = torch.tensor(self.df[aux_target_cols].values)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx].float(), self.targets[idx].float(), self.aux_targets[idx].float()


class NumeraiDataModule(LightningDataModule, ABC):
    def __init__(self, feature_set="small", sample_4th_era=True, aux_target_cols=None, batch_size=1000):
        super().__init__()
        if aux_target_cols is None:
            aux_target_cols = []
        self.save_hyperparameters()
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        download_current_data()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = NumeraiDataset("train",
                                             feature_set=self.hparams.feature_set,
                                             sample_4th_era=self.hparams.sample_4th_era,
                                             aux_target_cols=self.hparams.aux_target_cols)
            self.val_data = NumeraiDataset("val",
                                           feature_set=self.hparams.feature_set,
                                           sample_4th_era=self.hparams.sample_4th_era,
                                           aux_target_cols=self.hparams.aux_target_cols)
        elif stage == "test":
            self.test_data = NumeraiDataset("test",
                                            feature_set=self.hparams.feature_set,
                                            sample_4th_era=self.hparams.sample_4th_era,
                                            aux_target_cols=self.hparams.aux_target_cols)
        else:
            raise Exception(f"unsupported stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=NUM_WORKERS)
