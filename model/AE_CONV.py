from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import nn
from model.autoencoder import AutoEncoder
from model.ae import AE
nn = torch.nn


class AEConv(LightningModule, ABC):
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
        The idea is to go around the dimensions
        The dimensions should dip down in the middle and then output 1
        """

        self.encoder = []
        # First down
        self.conv1 = nn.Conv1d(38, 40, 1)
        self.max_pool1 = nn.MaxPool1d(1, stride=1)

        # Second Down
        self.conv2 = nn.Conv1d(40, 20, 1)
        self.max_pool2 = nn.MaxPool1d(1, stride=1)

        # Third Down
        self.conv3 = nn.Conv1d(20, 5, 1)
        self.max_pool3 = nn.MaxPool1d(1, stride=1)


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
        self.linear2 = nn.Linear(5,5)
        self.convdecode1 = nn.Conv1d(5, 5, 1)

        #upsample Convolution
        self.linear3 = nn.Linear(5, 20)
        self.convdecode2 = nn.Conv1d(20, 20, 1)


        # upsample Convolution

        self.linear4 = nn.Linear(5, 38)
        self.convdecode3 = nn.Conv1d(38, 38, 1)





    def forward(self, x):

        """
        Current problem: Sizes do not fit for the validation set
        """

        # print(x.size())
        #
        first, second = x.shape
        x = x.reshape(second,first)
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.convmid1(x)
        x = self.convmid2(x)
        x = self.convmid3(x)

        print('done with encoding')

        first, second = x.shape
        x = x.reshape(second, first)
        #
        x = self.linear1(x)
        print('first linear')

        first, second = x.shape
        x = x.reshape(second, first)

        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)



        print('done with mids')
        first, second = x.shape
        x = x.reshape(second, first)


        x = self.linear2(x)
        # first, second = x.shape
        # x = x.reshape(second, first)
        # print(x.shape)
        # x = self.convdecode1(x)

        # x = self.linear3(x)
        # x = self.convdecode2(x)
        # x = self.linear4(x)
        # x = self.convdecode3(x)













        return x

        # return up




