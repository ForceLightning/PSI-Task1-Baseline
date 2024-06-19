from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.args import DefaultArguments, ModelOpts
from utils.cuda import *


class TransformerTrajBbox(nn.Module):
    def __init__(self, args: DefaultArguments, model_opts: ModelOpts) -> None:
        super().__init__()

        self.args = args
        self.observe_length = args.observe_length
        self.predict_length = args.predict_length
        self.output_dim = model_opts["output_dim"]
        self.dropout = model_opts["dropout"]
        self.num_layers = model_opts["n_layers"]
        self.num_heads = model_opts.get("num_heads", 4)

        # Transformer encoder architecture
        self.trans_enc_in_dim = model_opts["enc_in_dim"]
        self.trans_enc_out_dim = model_opts["enc_out_dim"]

        # Transformer decoder architecture
        self.trans_dec_in_dim = model_opts.get("dec_in_emb_dim", self.trans_enc_out_dim)
        self.trans_dec_out_dim = model_opts["dec_out_dim"]
        self.output_dim = model_opts["output_dim"]

        self.src_mask = None

        # # Build encoder
        # encoder_layer = nn.TransformerEncoderLayer(
        #     self.trans_enc_in_dim,
        #     self.num_heads,
        #     dim_feedforward=self.trans_enc_out_dim,
        #     dropout=self.dropout,
        #     activation=F.mish,
        #     batch_first=True,
        # )
        # self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        #
        # # Build decoder
        # decoder_layer = nn.TransformerDecoderLayer(
        #     self.trans_dec_in_dim,
        #     self.num_heads,
        #     dim_feedforward=self.trans_dec_out_dim,
        #     dropout=self.dropout,
        #     activation=F.mish,
        #     batch_first=True,
        # )
        # self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)

        self.transformer = nn.Transformer(
            d_model=self.trans_enc_in_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.trans_dec_in_dim,
            dropout=self.dropout,
            activation=F.mish,
            batch_first=True,  # bs x ts x feature
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def forward(self, src: torch.Tensor, has_mask: bool = True) -> torch.Tensor:
        raise NotImplementedError
        # TODO: (chris) Finish writing up this method.

        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        #
        # y = self.transformer(
        #     src=src,
        # )
