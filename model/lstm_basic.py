import numpy as np
import pandas as pd
from numerapi import NumerAPI
import random
import torch
import math

import torch.nn.functional as F

nn = torch.nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden == None:
            self.hidden = (torch.zeros(1, 1, self.hidden_size),
                           torch.zeros(1, 1, self.hidden_size))
        else:
            self.hidden = hidden

        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1),
                                          self.hidden)

        linpreds = self.linear(lstm_out.view(len(x), -1))

        # Returns the last row - culmination of previous
        # Need to check if logically picking up the correct sequence

        # might need to use all previous for one of the values
        # this might not bring us anywhere
        return linpreds[-1], self.hidden