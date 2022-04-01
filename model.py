import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim


def evaluate(outputs):
    mae = torch.stack([x['loss'] for x in outputs]).mean()
    corrs = [x['corr'] for x in outputs]

    payout = torch.tensor(np.mean(np.divide(np.array(corrs), 0.2).clip(-1, 1)))
    spearman = torch.tensor(np.mean(corrs))
    std_corr = np.std(corrs)
    sharpe = spearman / std_corr

    return mae, payout, spearman, sharpe


class NumeraiModel(LightningModule):
    def __init__(self, model=None, feature_set="SMALL", aux_target_cols=None, dropout=0, initial_bn=False,
                 learning_rate=0.003, wd=5e-2):
        super().__init__()
        # Save for repeated runs, ignore the model itself
        if aux_target_cols is None:
            aux_target_cols = []
        self.save_hyperparameters(ignore="model")
        self.model = model or self.build_model()
        self.loss = nn.MSELoss()

    def build_model(self):
        layers = []

        if self.hparams.feature_set == "SMALL":
            num_features = [38, 20, 10]
        else:
            num_features = [1050, 200, 100]

        if self.hparams.initial_bn:
            layers.append(nn.BatchNorm1d(num_features[0]))

        for i in range(len(num_features) - 1):
            layers += [
                nn.Linear(num_features[i], num_features[i + 1], bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features[i + 1])
            ]

            if self.hparams.dropout > 0:
                layers.append(nn.Dropout(p=self.hparams.dropout))

        layers.append(nn.Linear(num_features[-1], 1 + len(self.hparams.aux_target_cols)))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, aux_targets = batch

        # Determine correlation of preds to targets
        preds = self.model(inputs)
        rank_pred = pd.Series(
            preds[:, 0].detach().cpu().numpy()).rank(pct=True, method='first')
        corr = np.corrcoef(targets.detach().cpu().numpy(), rank_pred)[0, 1]

        # Determine primary and auxiliary target loss
        loss = self.loss(preds[:, 0], targets)
        aux_loss = 0
        for i in range(len(self.hparams.aux_target_cols)):
            aux_loss += self.loss(preds[:, i + 1], aux_targets[:, i])

        self.log("train_loss", loss)
        self.log("train_loss_aux", aux_loss)
        self.log("train_corr", corr)

        return {'loss': loss + aux_loss, 'corr': corr}

    def training_epoch_end(self, outputs):
        mae, payout, spearman, sharpe = evaluate(outputs)

        self.log('train/mae', mae, on_step=False, on_epoch=True)
        self.log('train/payout', payout, on_step=False, on_epoch=True)
        self.log('train/spearman', spearman, on_step=False, on_epoch=True)
        self.log('train/sharpe', sharpe, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_nb):
        inputs, targets, _ = batch

        # Determine correlation of pred to targets
        preds = self.model(inputs)
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
