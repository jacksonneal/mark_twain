
from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import nn
nn = torch.nn


class AutoEncoder(LightningModule, ABC):

    def __init__(self, params):
        super().__init__()
        self.encoder = []
        self.decoder = []
        self.num_features = params.dimensions[0]
        self.out_dim = params.dimensions[2]
        self.hidden = params.dimensions[1]
        self.encoder += [nn.Linear(self.num_features, self.hidden)]
        self.encoder += [nn.BatchNorm1d(self.hidden)]
        self.encoder += [nn.SiLU()]

        self.decoder += [nn.Linear(self.hidden, self.hidden)]
        self.decoder += [nn.BatchNorm1d(self.hidden)]
        self.decoder += [nn.SiLU()]
        self.decoder += [nn.Dropout(p=.2)]
        self.decoder += [nn.Linear(self.hidden, self.out_dim)]
        # self.dropout = dropout



    def forward(self, x):



        batch1 = nn.BatchNorm1d(self.num_features)
        auto = batch1(x)



        ## Guasssian Noise, Can add Param for variance
        variance = .1
        auto = auto + (variance ** 0.5) * torch.randn(auto.shape)

        encode = nn.Sequential(*self.encoder)
        auto = encode(auto)
        decode = nn.Sequential(*self.decoder)
        auto = decode(auto)


        ##TODO: 3: Prepare Input for number of hidden
        ## This process will be done after we have the regular encoder - decoder working

        # x = tf.keras.layers.Concatenate()([x0, encoder])
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(dropout_rates[3])(x)







        return auto

    #


## Basic Testing

## grab the data





