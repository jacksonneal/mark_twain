import matplotlib
import numpy as np
import pandas as pd
from numerapi import NumerAPI
import random
import torch
import math
from tqdm.notebook import tqdm
nn = torch.nn


class NN_Basic(nn.Module):
    def __init__(self, num_feature, hidden1, hidden2, out_dim):
        super(NN_Basic, self).__init__()

        self.layer_1 = nn.Linear(num_feature, hidden1)
        self.layer_2 = nn.Linear(hidden1, hidden2)
        self.layer_3 = nn.Linear(hidden2, out_dim)
        self.layer_out = nn.Linear(out_dim, out_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(hidden1)
        self.batchnorm2 = nn.BatchNorm1d(hidden2)
        self.batchnorm3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        return x