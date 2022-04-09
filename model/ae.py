from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import nn

from model.gaussian_noise import GaussianNoise


class AE(LightningModule, ABC):

    def __init__(self, params):
        super().__init__()
        dimensions = params.dimensions

        encoder_layers = [nn.Linear(dimensions[0], dimensions[1]), nn.BatchNorm1d(dimensions[1]),
                          nn.SiLU(inplace=True)]
        if params.dropout > 0:
            encoder_layers.append(nn.Dropout(p=params.dropout))

        decoder_layers = []
        if params.dropout > 0:
            decoder_layers.append(nn.Dropout(p=params.dropout))
        decoder_layers.append(nn.Linear(dimensions[1], dimensions[0]))

        ae_layers = [nn.Linear(dimensions[0], dimensions[2]), nn.BatchNorm1d(dimensions[2]), nn.SiLU(inplace=True)]
        if params.dropout > 0:
            ae_layers.append(nn.Dropout(p=params.dropout))
        ae_layers.append(nn.Linear(dimensions[2], 1 + len(params.aux_target_cols)))
        ae_layers.append(nn.Sigmoid())

        mlp_layers = [nn.Linear(dimensions[0] + dimensions[1], dimensions[3]), nn.BatchNorm1d(dimensions[3]),
                      nn.SiLU(inplace=True)]
        if params.dropout > 0:
            mlp_layers.append(nn.Dropout(p=params.dropout))
        for i in range(3, len(dimensions) - 1):
            mlp_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            mlp_layers.append(nn.BatchNorm1d(dimensions[i + 1]))
            mlp_layers.append(nn.SiLU(inplace=True))
            if params.dropout > 0:
                mlp_layers.append(nn.Dropout(p=params.dropout))
        mlp_layers.append(nn.Linear(dimensions[-1], 1 + len(params.aux_target_cols)))
        mlp_layers.append(nn.Sigmoid())

        if params.initial_bn:
            self.initial_bn = nn.BatchNorm1d(self.num_features)
        else:
            self.initial_bn = None
        self.num_features = dimensions[0]
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.ae = nn.Sequential(*ae_layers)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        if self.initial_bn is not None:
            x = self.initial_bn(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        ae_out = self.ae(decoded)
        mlp_out = self.mlp(torch.cat((x, encoded), dim=1).to(self.device))
        return decoded, ae_out, mlp_out
