from abc import ABC
from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
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
        self.num_features = None

    def prepare_data(self) -> None:
        preprocessing.download_current_data()
        preprocessing.create_feature_sets()
        self.num_features = preprocessing.get_num_features(feature_set=self.hparams.feature_set,
                                                           sample_4th_era=self.hparams.sample_4th_era,
                                                           pca=self.hparams.pca)

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = NumeraiDataset("train",
                                             feature_set=self.hparams.feature_set,
                                             sample_4th_era=self.hparams.sample_4th_era,
                                             aux_target_cols=self.hparams.aux_target_cols,
                                             pca=self.hparams.pca)
            val_pca = self.num_features if self.hparams.pca is not None else None
            self.val_data = NumeraiDataset("val",
                                           feature_set=self.hparams.feature_set,
                                           sample_4th_era=self.hparams.sample_4th_era,
                                           aux_target_cols=self.hparams.aux_target_cols,
                                           pca=val_pca)
        elif stage == "predict":
            test_pca = self.num_features if self.hparams.pca is not None else None
            self.test_data = NumeraiDataset("test",
                                            feature_set=self.hparams.feature_set,
                                            sample_4th_era=False,
                                            aux_target_cols=[],
                                            pca=test_pca)
        else:
            raise Exception(f"unsupported stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=self.num_workers)
