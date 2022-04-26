from typing import Optional

import torch
from torch.utils.data import Dataset

from numerai.data import preprocessing
from numerai.definitions import TARGET_COL


class NumeraiDataset(Dataset):
    def __init__(self, name="train", feature_set="small", sample_4th_era=True, aux_target_cols=None,
                 pca: Optional[float] = None):
        super().__init__()
        if aux_target_cols is None:
            aux_target_cols = []
        self.df = preprocessing.load_data(name,
                                          feature_set=feature_set,
                                          sample_4th_era=sample_4th_era,
                                          aux_target_cols=aux_target_cols,
                                          pca=pca)

        # ensure auxiliary targets have values
        for target_aux_col in aux_target_cols:
            self.df.loc[:, target_aux_col].fillna(0.5, inplace=True)

        # extract feature col names
        feature_cols = [c for c in self.df if c.startswith("feature_")]

        self.num_features = len(feature_cols)
        self.inputs = torch.tensor(self.df[feature_cols].values)
        self.targets = torch.tensor(self.df[TARGET_COL].values)
        self.aux_targets = torch.tensor(self.df[aux_target_cols].values)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx].float(), self.targets[idx].float(), self.aux_targets[idx].float()
