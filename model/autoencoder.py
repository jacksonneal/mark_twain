
from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import nn
nn = torch.nn


class AutoEncoder(LightningModule, ABC):

    def __init__(self, params):
        super().__init__()
        dimensions = params.dimensions
        self.encoder = []
        self.decoder = []
        self.num_features = dimensions[0]
        self.out_dim = dimensions[2]
        self.hidden = dimensions[1]
        self.encoder += [nn.Linear(self.num_features, self.hidden)]
        self.encoder += [nn.BatchNorm1d(self.hidden)]
        self.encoder += [nn.SiLU()]

        self.decoder += [nn.Linear(self.hidden, self.hidden)]
        self.decoder += [nn.BatchNorm1d(self.hidden)]
        self.decoder += [nn.SiLU()]
        self.decoder += [nn.Dropout(p=.2)]
        self.decoder += [nn.Linear(self.hidden, self.out_dim)]
        # self.dropout = dropout
        self.encode = nn.Sequential(*self.encoder)
        self.decode = nn.Sequential(*self.decoder)



    def forward(self, x):



        batch1 = nn.BatchNorm1d(self.num_features)
        auto = batch1(x)



        ## Guasssian Noise, Can add Param for variance
        variance = .1
        auto = auto + (variance ** 0.5) * torch.randn(auto.shape)

        encoded = self.encode(auto)
        auto = self.decode(encoded)

        return auto






