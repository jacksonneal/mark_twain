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
        self.num_feats = self.dimensions[0]
        self.encoder = []
        # First down
        self.conv1 = nn.Conv1d(self.num_feats, 40, 1)
        self.max_pool1 = nn.MaxPool1d(1, stride=1)

        # Second Down
        self.conv2 = nn.Conv1d(40, 20, 1)
        self.max_pool2 = nn.MaxPool1d(1, stride=1)

        # Third Down
        self.conv3 = nn.Conv1d(20, 5, 1)
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
        self.linear2 = nn.Linear(5,10)
        self.convdecode1 = nn.Conv1d(10, 10, 1)

        #upsample Convolution
        self.linear3 = nn.Linear(10, 20)
        self.convdecode2 = nn.Conv1d(20, 20, 1)


        # upsample Convolution

        self.linear4 = nn.Linear(20, self.num_feats)
        self.convdecode3 = nn.Conv1d(self.num_feats, self.num_feats, 1)





    def forward(self, x):

        """
        Current problem: Sizes do not fit for the validation set
        """

        # print(x.size())
        #
        # first, second = x.shape
        # x = x.reshape(second,first)
        x = x.unsqueeze(dim=2)
        x = self.conv1(x)
        # x = x.squeeze()
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.convmid1(x)
        x = self.convmid2(x)
        x = self.convmid3(x)

        print('done with encoding')

        x = x.squeeze()

        # first, second = x.shape
        # x = x.reshape(second, first)
        #
        x = self.linear1(x)
        print('first linear')



        x = x.unsqueeze(dim=2)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)



        print('done with mids')
        x = x.squeeze()
        # first, second = x.shape
        # x = x.reshape(second, first)


        x = self.linear2(x)

        x = x.unsqueeze(dim=2)
        # first, second = x.shape
        # x = x.reshape(second,first)

        x = self.convdecode1(x)
        # print('FINE')

        # first, second = x.shape
        # x = x.reshape(second, first)
        x = x.squeeze()
        x = self.linear3(x)

        print('linear 3 good')

        # first, second = x.shape
        # x = x.reshape(second, first)
        x = x.unsqueeze(dim=2)
        x = self.convdecode2(x)

        # first, second = x.shape
        # x = x.reshape(second, first)
        x = x.squeeze()
        x = self.linear4(x)
        # first, second = x.shape
        # x = x.reshape(second, first)
        # x = self.convdecode3(x)

    # TODO: Figure out why sweep doesnt work
#Implement Decoder
        # make modular

        return x

        # return up




