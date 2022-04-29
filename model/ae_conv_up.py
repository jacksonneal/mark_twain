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

        dim1 = 30
        dim2 = 10
        # decoding
        self.conv_out = nn.Conv1d(dim1, dim1, 1)
        self.linear = nn.Linear(dim1, features)

        #TODO: Define Kernel
        #TODO: Define Stride
            #Conv
            # Max pooling
        # Encoder
            # 3 Convolutional Layers
            # 3 max pooling layers
        self.conv1 = nn.Conv1d(features, dim1, 1, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=5, stride=1)

        self.batch_norm1 = nn.BatchNorm1d(dim1)
        self.silu = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv1d(dim1, dim2,1, stride=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=5,stride=1)

        self.batch_norm2 = nn.BatchNorm1d(dim2)

        self.conv3 = nn.Conv1d(dim2, 5, 1, stride=1)
        self.batch_norm3 = nn.Conv1d(5)
        self.max_pool3 = nn.MaxPool1d(kernel_size=5, stride=1)

        # middle
        # 3 middle
        # 1 up sample
        # 3 more middle

        self.convmid1 = nn.Conv1d(5,2, 1)
        self.convmid2 = nn.Conv1d(2, 2, 1)
        self.convmid3 = nn.Conv1d(2, 2, 1)


        # Decoder
            # upsample
            #3 convolutional

        self.mid1 = nn.Conv1d(5, 5, 1)
        self.mid2 = nn.Conv1d(5, 5, 1)
        self.mid3 = nn.Conv1d(5,5, 1)

        self.convdecode1 = nn.Conv1d(5,dim2,1)
        self.convdecode2 = nn.Conv1d(dim2, dim1, 1)
        self.convdecode3 = nn.Conv1d(dim1, dim1, 1)

        # One Linear out
        # also switch out for base
        self.linear = nn.Linear(dim1, self.features)

        self.dropout = nn.Dropout(params.dropout)



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
        target = x.shape[0]
        # First Convolution
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 2, 1)

        encode_con = self.conv1(x)
        encode_pool = self.batch_norm1(encode_con)
        encode_con = self.silu(encode_pool)
        encode_con = encode_con.squeeze()
        encode_con = encode_con.transpose(0, 1)
        encode_con = encode_con.unsqueeze(dim=1)
        encode_pool = self.max_pool1(encode_con)

        # encode_pool = encode_pool.squeeze()



        # x = encode_pool.unsqueeze(dim=1)

        encode_pool = encode_pool.permute(2, 0,1)

        target3 = encode_pool.shape[0]
        encode_con = self.conv2(encode_pool)
        encode_con = self.batch_norm2(encode_con)
        encode_con = self.silu(encode_con)
        encode_con = encode_con.squeeze()
        encode_con = encode_con.transpose(0, 1)
        encode_con = encode_con.unsqueeze(dim=1)
        encode_pool = self.max_pool2(encode_con)
        # encode_pool = encode_pool.squeeze()



        encode_pool = encode_pool.permute(2, 0,1)

        target2 = encode_pool.shape[0]
        encode_con = self.conv3(encode_pool)
        encode_con = self.batch_norm3(encode_con)
        encode_con = self.silu(encode_con)
        encode_con = encode_con.squeeze()
        encode_con = encode_con.transpose(0, 1)
        encode_con = encode_con.unsqueeze(dim=1)
        encode_pool = self.max_pool3(encode_con)

        #### first drop out layer
        # encode_pool = self.dropout(encode_pool)

        encode_pool = encode_pool.permute(2,0,1)

        ## First Target
        target1 = encode_pool.shape[0]
        x = self.convmid1(encode_pool)
        x = self.convmid2(x)
        x = self.convmid3(x)

        # Must Squeeze for up sample

        x = x.squeeze()


        pool_shape = max_pool_shape(x.transpose(0, 1), 5, 1)

        scale = target1 / pool_shape

        unsqueezed = encode_pool.unsqueeze(dim=2)

        unsqueezed = unsqueezed.permute(1,2,3,0)
        up = nn.Upsample(scale_factor=scale)
        up_scaled = up(unsqueezed)


        up_scaled = up_scaled.squeeze()


        x = up_scaled.unsqueeze(dim=2)
        x= x.permute(1,0,2)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)

        x = x.squeeze()

        pool_shape = max_pool_shape(x.transpose(0, 1), 5, 1)
        scale = target2 / pool_shape


        unsqueezed = encode_pool.unsqueeze(dim=2)

        unsqueezed = unsqueezed.permute(1,2,3,0)
        up = nn.Upsample(scale_factor=scale)
        up_scaled = up(unsqueezed)
        up_scaled = up_scaled.squeeze()
        up_scaled = up_scaled.unsqueeze(dim=2)
        # print(up_scaled)
        up_scaled = up_scaled.permute(2,0,1)
        x = self.convdecode1(up_scaled)
        # print(x.shape)
        x = x.squeeze()

        ### Next Upscale -

        # pool_shape = max_pool_shape(x.transpose(0, 1), 5, 1)
        scale = target3 / x.shape[1]
        # print('POOOLL SHAPE', pool_shape)
        # print('THIS IS SCALE')
        # print(scale)
        #
        unsqueezed = x.unsqueeze(dim=2)
        # print(unsqueezed.shape)
        unsqueezed = unsqueezed.permute(0,2,1)
        up = nn.Upsample(scale_factor=scale)
        up_scaled = up(unsqueezed)
        up_scaled = up_scaled.squeeze()
        #
        # print(target3)
        # print(up_scaled.shape)
        #
        # # up_scaled = up_scaled.transpose(0,1)
        up_scaled = up_scaled.unsqueeze(dim=2)
        # print(up_scaled)
        up_scaled = up_scaled.permute(2,0,1)
        x = self.convdecode2(up_scaled)
        x = x.squeeze()

        scale = target / x.shape[1]
        # print('POOOLL SHAPE', pool_shape)
        # print('THIS IS SCALE')
        # print(scale)
        #
        unsqueezed = x.unsqueeze(dim=2)
        # print(unsqueezed.shape)
        unsqueezed = unsqueezed.permute(0,2,1)

        up = nn.Upsample(scale_factor=scale)
        up_scaled = up(unsqueezed)
        up_scaled = up_scaled.squeeze()
        #
        # print(target)
        # print(up_scaled.shape)
        #
        # # up_scaled = up_scaled.transpose(0,1)
        up_scaled = up_scaled.unsqueeze(dim=2)
        # print(up_scaled)
        up_scaled = up_scaled.permute(2,0,1)
        x = self.convdecode3(up_scaled)
        x = x.squeeze()


        x = x.transpose(0,1)
        out = self.linear(x)

        return out