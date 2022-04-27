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
        self.num_features = params.dimensions[0]
        # self.ae = AutoEncoder(params)
        self.ae = AE(params)
        self.params = params

        """
        
        Here is the deal I am not sure how to change the dimensions on this one. 
        It would be great if I could change the yaml files but I am struggling with that. 
        
        
        
        """

    def forward(self, x):

        # batch1 = nn.BatchNorm1d(self.num_features)
        decoded, out, mlp = self.ae(x)



        out = 0

        return out




