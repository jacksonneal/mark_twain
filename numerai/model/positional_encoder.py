import math
from abc import ABC
import torch
from pytorch_lightning import LightningModule


class PositionalEncoder(LightningModule, ABC):
    """
    A module that adds positional encoding to each of the token's features.
    So that the Transformer is position aware.
    """

    def __init__(self, input_dim: int, max_len: int = 10000):
        """
        Inputs:
        - input_dim: Input dimension about the features for each token
        - max_len: The maximum sequence length
        """
        super().__init__()
        self.input_dim = input_dim
        self.max_len = max_len

    def forward(self, x):
        """
        Compute the positional encoding and add it to x.

        Input:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
          and C is the channel dimension

        Return:
        - x: Tensor of the shape BxLxC, with the positional encoding added to the input
        """
        seq_len = x.shape[1]
        input_dim = x.shape[2]

        pe = None
        ###########################################################################
        # TODO: Compute the positional encoding                                   #
        # Check Section 3.5 for the definition (https://arxiv.org/pdf/1706.03762.pdf)
        #                                                                         #
        # It's a bit messy, but the definition is provided for your here for your #
        # convenience (in LaTex).                                                 #
        # PE_{(pos,2i)} = sin(pos / 10000^{2i/\dmodel})                           #
        # PE_{(pos,2i+1)} = cos(pos / 10000^{2i/\dmodel})                         #
        #                                                                         #
        # You should replace 10000 with max_len here.
        ###########################################################################
        pe = torch.zeros(seq_len, input_dim)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        den = torch.exp(torch.arange(0, input_dim, 2, dtype=torch.float) * (-math.log(self.max_len) / input_dim))
        pe[:, 0::2] = torch.sin(pos.float() * den)
        pe[:, 1::2] = torch.cos(pos.float() * den)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        x = x + pe.to(x.device)
        return x
