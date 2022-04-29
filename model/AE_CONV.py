from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import nn
from model.autoencoder import AutoEncoder
from model.ae import AE
from model.base import Base
nn = torch.nn


class AEConv(LightningModule, ABC):
    def __init__(self, params):
        super().__init__()
        print(params)
        # setting up - dimensions
        self.dimensions = params.dimensions
        self.base = Base(params)
        # self.num_features = params.dimensions[0]
        #
        # layers = []
        # for i in range(3, len(self.dimensions) - 3, 3):

        """
        The idea is to go around the dimensions
        The dimensions should dip down in the middle and then output 1
        """

        def dimension_calc(input, kernel, stride=1, padding=0, dilation=1):
            shape = input.shape
            last_dim = shape[-1]

            out = ((last_dim + 2 * padding - dilation*(kernel - 1) -1 ) / stride) + 1

            return out

        #TODO: Cite this work
        def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
            from math import floor
            if type(kernel_size) is not tuple:
                kernel_size = (kernel_size, kernel_size)
            w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
            return w

        kernel_size = 4

        self.num_feats = self.dimensions[0]

        self.dropout = nn.Dropout(p=params.dropout)
        self.encoder = []
        # First down
        self.conv1 = nn.Conv1d(self.num_feats, 40, 1)
        self.batch_norm1 = nn.BatchNorm1d(40)
        self.silu = nn.SiLU(inplace=True)

        self.max_pool1 = nn.MaxPool1d(1, stride=1)


        # Second Down
        self.conv2 = nn.Conv1d(40, 20, 1)
        self.batch_norm2 = nn.BatchNorm1d(20)
        self.max_pool2 = nn.MaxPool1d(1, stride=1)

        # Third Down
        self.conv3 = nn.Conv1d(20, 5, 1)
        self.batch_norm3 = nn.BatchNorm1d(5)
        self.max_pool3 = nn.MaxPool1d(1, stride=1)

        ## Middle Portion
        # Three Convolutions
        self.convmid1 = nn.Conv1d(5,2, 1)
        self.convmid2 = nn.Conv1d(2, 2, 1)
        self.convmid3 = nn.Conv1d(2, 2, 1)


        # Upsample + 3 Convolutions

        self.linear1 = nn.Linear(2, 5)


        self.mid1 = nn.Conv1d(5, 5, 1)
        self.mid2 = nn.Conv1d(5, 5, 1)
        self.mid3 = nn.Conv1d(5,5, 1)

        # upsample Convolution
        self.linear2 = nn.Linear(5,20)
        self.convdecode1 = nn.Conv1d(20, 40, 1)

        #upsample Convolution
        self.linear3 = nn.Linear(40, 40)
        self.convdecode2 = nn.Conv1d(40, 40, 1)


        # upsample Convolution

        self.linear4 = nn.Linear(40, self.num_feats)
        self.convdecode3 = nn.Conv1d(self.num_feats, self.num_feats, 1)

        self.rel = nn.ReLU()



    def forward(self, x):

        """
        Current problem: Sizes do not fit for the validation set
        """

        #Encoding

        x = x.unsqueeze(dim=2)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.silu(x)
        x = self.dropout(x)


        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.silu(x)
        # x = self.dropout(x)
        x = self.max_pool2(x)
        # x = m(x)
        x = self.conv3(x)
        # Notice: Adding batch Norm to the  third convolution
        # # this actually reduced the correlation
        # x = self.batch_norm3(x)
        # x = self.silu(x)
        x = self.max_pool3(x)

        # middle Section
        x = self.convmid1(x)
        x = self.convmid2(x)
        x = self.convmid3(x)
        # x = m(x)



        x = x.squeeze()

        x = self.linear1(x)
        # x = m(x)

        x = x.unsqueeze(dim=2)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)
        # x = m(x)

        # Decoding

        x = x.squeeze()
        x = self.linear2(x)
        x = x.unsqueeze(dim=2)
        x = self.convdecode1(x)
        x = x.squeeze()
        x = self.linear3(x)
        x = x.unsqueeze(dim=2)
        x = self.convdecode2(x)
        x = x.squeeze()
        x = self.linear4(x)


        return x


# add silu and dropout and batch nor


