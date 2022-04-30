from abc import ABC
import torch
from pytorch_lightning import LightningModule
from torch import nn
from numerai.model.positional_encoder import PositionalEncoder


class TransformerMLP(LightningModule, ABC):

    def __init__(self, params):
        super().__init__()
        if params.initial_bn:
            self.initial_bn = nn.BatchNorm1d(params.num_features)
        else:
            self.initial_bn = None
        self.pos_encoder = PositionalEncoder(params.num_features)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1,
                                                   dropout=params.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, params.num_enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=1, nhead=1,
                                                   dropout=params.dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, params.num_dec_layers)
        dimensions = params.dimensions
        ae_layers = [nn.Linear(params.num_features, dimensions[0]), nn.BatchNorm1d(dimensions[0]),
                     nn.SiLU(inplace=True)]
        if params.dropout > 0:
            ae_layers.append(nn.Dropout(p=params.dropout))
        ae_layers.append(nn.Linear(dimensions[0], 1 + len(params.aux_target_cols)))
        ae_layers.append(nn.Sigmoid())
        self.ae = nn.Sequential(*ae_layers)
        mlp_layers = [nn.Linear(params.num_features + params.num_features, dimensions[1]),
                      nn.BatchNorm1d(dimensions[1]),
                      nn.SiLU(inplace=True)]
        if params.dropout > 0:
            mlp_layers.append(nn.Dropout(p=params.dropout))
        for i in range(1, len(dimensions) - 1):
            mlp_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            mlp_layers.append(nn.BatchNorm1d(dimensions[i + 1]))
            mlp_layers.append(nn.SiLU(inplace=True))
            if params.dropout > 0:
                mlp_layers.append(nn.Dropout(p=params.dropout))
        mlp_layers.append(nn.Linear(dimensions[-1], 1 + len(params.aux_target_cols)))
        mlp_layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        if self.initial_bn is not None:
            x = self.initial_bn(x)
        # print(x.shape)
        x = x.unsqueeze(2)
        # print(x.shape)
        src = x
        # src_pos = self.pos_encoder(src)
        src_pos = src
        tgt_pos = src_pos
        encoded = self.transformer_encoder(src_pos)
        decoded = self.transformer_decoder(tgt_pos, src_pos)
        # print(decoded.shape)
        decoded = decoded.squeeze()
        # print(decoded.shape)
        ae_out = self.ae(decoded)
        src_pos = src_pos.squeeze()
        encoded = encoded.squeeze()
        # print(f"src_pos.shape {src_pos.shape}")
        # print(f"encoded.shape {encoded.shape}")
        mlp_out = self.mlp(torch.cat((src_pos, encoded), dim=1).to(self.device))
        return decoded, ae_out, mlp_out
