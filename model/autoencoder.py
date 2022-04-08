import matplotlib
import numpy as np
import pandas as pd
from numerapi import NumerAPI
import random
import torch
import math
from tqdm.notebook import tqdm
nn = torch.nn


class AutoEncoder(nn.Module):

    def __init__(self, num_features, hidden,  out_dim, dropout):
        self.encoder = []
        self.decoder = []
        self.num_features = num_features
        self.out_dim = out_dim
        self.hidden = hidden
        self.buildEncoderDecoder()
        self.dropout = dropout

    def buildEncoderDecoder(self):

        self.encoder += [nn.Linear(self.num_features, self.hidden)]
        self.encoder += [nn.BatchNorm1d(self.hidden)]
        self.encoder += [nn.SiLU()]


        self.decoder += [nn.Linear(self.hidden, self.hidden)]
        self.decoder += [nn.BatchNorm1d(self.hidden)]
        self.decoder += [nn.SiLU()]
        self.decoder += [nn.Dropout(p=.2)]
        self.decoder += [nn.Linear(self.hidden, self.out_dim)]
        ## TODO: DropOut to be added in the forward


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

        print(auto.shape)





        return auto

    #


## Basic Testing

## grab the data

input = torch.randn(20, 100)



auto = AutoEncoder(100, 4, 1, .2)
final = auto.forward(input)



