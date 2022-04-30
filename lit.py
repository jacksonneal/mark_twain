from abc import ABC

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim

from model.model_factory import build_model


def evaluate(outputs):
    mae = torch.stack([x['loss'] for x in outputs]).mean()
    corrs = [x['corr'] for x in outputs]

    payout = torch.tensor(np.mean(np.divide(np.array(corrs), 0.2).clip(-1, 1)))
    spearman = torch.tensor(np.mean(corrs))
    std_corr = np.std(corrs)
    sharpe = spearman / std_corr

    return mae, payout, spearman, sharpe


class NumeraiLit(LightningModule, ABC):
    def __init__(self, model=None, model_name=None, feature_set="small", dimensions=None, aux_target_cols=None,
                 dropout=0, initial_bn=False, learning_rate=0.003, wd=5e-2):
        super().__init__()
        if dimensions is None:
            dimensions = [38, 20, 10]
        if aux_target_cols is None:
            aux_target_cols = []
        # Save for repeated runs, ignore the model itself
        self.save_hyperparameters(ignore="model")
        self.model = model or build_model(self.hparams)
        self.loss = nn.MSELoss()
        self.ae_architecture = model_name == "AE"
        self.cae = model_name == 'CAE'

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, aux_targets = batch

        # Determine correlation of preds to targets
        preds = self.model(inputs)
        if self.ae_architecture or self.cae:
            decoded, ae_out, preds = preds
        else:
            decoded = None
            ae_out = None
        rank_pred = pd.Series(
            preds[:, 0].detach().cpu().numpy()).rank(pct=True, method='first')
        corr = np.corrcoef(targets.detach().cpu().numpy(), rank_pred)[0, 1]

        # Determine primary and auxiliary target loss
        loss = self.loss(preds[:, 0], targets)
        if self.ae_architecture or self.cae:
            loss += self.loss(ae_out[:, 0], targets)
            loss += self.loss(decoded, inputs)
        aux_loss = 0.0
        for i in range(len(self.hparams.aux_target_cols)):
            aux_loss += self.loss(preds[:, i + 1], aux_targets[:, i])
            if self.ae_architecture:
                aux_loss += self.loss(ae_out[:, i + 1], aux_targets[:, i])

        self.log("train_loss", loss)
        self.log("train_loss_aux", aux_loss)
        self.log("train_corr", corr)

        return {'loss': loss + aux_loss, 'corr': corr}

    def training_epoch_end(self, outputs):
        mae, payout, spearman, sharpe = evaluate(outputs)

        self.log('train/mae', mae)
        self.log('train/payout', payout)
        self.log('train/spearman', spearman)
        self.log('train/sharpe', sharpe)

    def validation_step(self, batch, batch_nb):
        inputs, targets, _ = batch

        # Determine correlation of pred to targets
        preds = self.model(inputs)
        if self.ae_architecture or self.cae:
            _, _, preds = preds
        rank_pred = pd.Series(preds[:, 0].cpu()).rank(pct=True, method='first')
        corr = np.corrcoef(targets.cpu(), rank_pred)[0, 1]

        # Determine primary target loss
        loss = self.loss(preds[:, 0], targets)

        self.log("val_loss", loss)
        self.log("val_corr", corr)

        return {'loss': loss, 'corr': corr}

    def validation_epoch_end(self, outputs):
        mae, payout, spearman, sharpe = evaluate(outputs)

        self.log('val/mae', mae)
        self.log('val/payout', payout)
        self.log('val/spearman', spearman)
        self.log('val/sharpe', sharpe)

    def configure_optimizers(self):
        return optim.AdamW(
            self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
