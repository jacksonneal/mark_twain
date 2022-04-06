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

    def __init__(self, num_features, hidden,  out_dim):
        self.encoder = []
        self.decoder = []
        self.num_features = num_features
        self.out_dim = out_dim
        self.hidden = hidden
        self.buildEncoderDecoder()

    def buildEncoderDecoder(self):

        ## gaussian Noise can be placed

        ## TODO: Normalize in the forward function

        ## TODO: Add Gaussian Noise
        ## Please Note that the Gaussian Noise can be added in the forward

        ## Dense is equivalent
        self.encoder += [nn.Linear(self.num_features, self.hidden)]
        self.encoder += [nn.BatchNorm1d(self.hidden)]
        self.encoder += [nn.SiLU()]


        self.decoder += [nn.Linear(self.hidden, self.hidden)]
        self.decoder += [nn.BatchNorm1d(self.hidden)]
        self.decoder += [nn.SilU()]

        self.decoder = [nn.Linear(self.hidden, self.out_dim)]
        ## TODO: DropOut to be added in the forward


    def forward(self, x):

        ##TODO: 1. Batch Norm
        ## TODO 2: Gaussian Noise

        ##TODO: 3: Prepare Input for number of hidden

        # x = tf.keras.layers.Concatenate()([x0, encoder])
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(dropout_rates[3])(x)





        pass 

    #

