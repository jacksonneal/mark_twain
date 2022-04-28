from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import nn
from model.autoencoder import AutoEncoder
from model.ae import AE
nn = torch.nn


class AEUP(LightningModule, ABC):
    def __init__(self, params):
        super().__init__()
        print(params)
        # setting up - dimensions
        self.dimensions = params.dimensions
        # self.num_features = params.dimensions[0]
        #
        # layers = []
        # for i in range(3, len(self.dimensions) - 3, 3):

        """
        Very simple auto encoder decoder with upsampling
        """

        # encoding
        features = params.dimensions[0]
        self.features = features
        # starting with kernel of 1 for simplicity
        #TODO: Adjust kernel size
        self.conv1 = nn.Conv1d(features, 40, 1, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=5, stride=1)

        # decoding
        self.conv_out = nn.Conv1d(40, 40, 1)
        self.linear = nn.Linear(40, features)



    def forward(self, x):
        print('*'*50)
        print(x.shape)

        print('*' * 50)

        """
        Current problem: Sizes do not fit for the validation set
        """

        def max_pool_shape(input, kernel, stride=1):
            return int((input.shape[1] + 2 * 0 - 1 * (kernel - 1) - 1) / stride + 1)

        # encoding step
        # x = x.transpose(0,1)
        x = x.unsqueeze(dim=2)
        encode_con = self.conv1(x)

        print('ENCODED CON')
        print(encode_con.shape)
        encode_pool = self.max_pool1(encode_con)
        x = x.squeeze()

        # upscaling step

        pool_shape = max_pool_shape(x, 5, 1)
        target = x.shape[1]
        scale = target / pool_shape
        unsqueezed = encode_pool.unsqueeze(dim=2)
        unsqueezed = unsqueezed.permute(0, 2, 1)
        up = nn.Upsample(scale_factor=scale)
        up_scaled = up(unsqueezed)
        up_scaled = up_scaled.permute(0, 2, 1)
        up_scaled = up_scaled.squeeze()

        ## decoding
        out = self.conv_out(up_scaled)
        out = out.transpose(0, 1)
        out = self.linear(out)

        return out