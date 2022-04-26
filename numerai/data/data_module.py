from abc import ABC
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from numerai.data import preprocessing
from numerai.data.dataset import NumeraiDataset


class NumeraiDataModule(LightningDataModule, ABC):
    def __init__(self, feature_set="small", sample_4th_era=True, aux_target_cols=None, batch_size=1000, num_workers=1,
                 pca: Optional[float] = None):
        super().__init__()
        if aux_target_cols is None:
            aux_target_cols = []
        self.save_hyperparameters(ignore="num_workers")
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        preprocessing.download_current_data()
        preprocessing.create_feature_sets()

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = NumeraiDataset("train",
                                             feature_set=self.hparams.feature_set,
                                             sample_4th_era=self.hparams.sample_4th_era,
                                             aux_target_cols=self.hparams.aux_target_cols,
                                             pca=self.hparams.pca)
            self.val_data = NumeraiDataset("val",
                                           feature_set=self.hparams.feature_set,
                                           sample_4th_era=self.hparams.sample_4th_era,
                                           aux_target_cols=self.hparams.aux_target_cols,
                                           pca=self.hparams.pca)
        elif stage == "test":
            self.test_data = NumeraiDataset("test",
                                            feature_set=self.hparams.feature_set,
                                            sample_4th_era=self.hparams.sample_4th_era,
                                            aux_target_cols=self.hparams.aux_target_cols,
                                            pca=self.hparams.pca)
        else:
            raise Exception(f"unsupported stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=self.num_workers)
