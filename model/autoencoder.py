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
        self.decoder += [nn.SilU()]
        self.decoder += [nn.Dropout(self.dropout)]
        self.decoder += [nn.Linear(self.hidden, self.out_dim)]
        ## TODO: DropOut to be added in the forward


    def forward(self, x):



        auto = nn.BatchNorm1d(x)

        ## TODO 2: Gaussian Noise


        ## Noise example Not sure if I will use

        sampled_noise = self.noise.repeat(*x.size()).normal_()
        auto = auto + sampled_noise

        encode = nn.Sequential(*self.encoder)

        auto = encode(auto)

        decode = nn.Sequential(*self.decoder)

        out = decode(auto)


        ##TODO: 3: Prepare Input for number of hidden
        ## This process will be done after we have the regular encoder - decoder working

        # x = tf.keras.layers.Concatenate()([x0, encoder])
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(dropout_rates[3])(x)





        return out

    #

