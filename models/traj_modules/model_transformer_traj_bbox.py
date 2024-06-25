from __future__ import annotations
from typing import Any, overload
from torch.nn.modules import Transformer
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
)
from transformers.modeling_outputs import SampleTSPredictionOutput
from transformers.models.time_series_transformer.modeling_time_series_transformer import (
    Seq2SeqTSPredictionOutput,
)
from typing_extensions import override

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig

from data.custom_dataset import T_intentBatch
from utils.args import DefaultArguments, ModelOpts
from utils.cuda import *


# class TransformerTrajBbox(nn.Module):
#     def __init__(self, args: DefaultArguments, model_opts: ModelOpts) -> None:
#         super().__init__()
#
#         self.args = args
#         self.observe_length = args.observe_length
#         self.predict_length = args.predict_length
#         self.output_dim = model_opts["output_dim"]
#         self.dropout = model_opts["dropout"]
#         self.num_layers = model_opts["n_layers"]
#         self.num_heads = model_opts.get("num_heads", 4)
#
#         # Transformer encoder architecture
#         self.trans_enc_in_dim = model_opts["enc_in_dim"]
#         self.trans_enc_out_dim = model_opts["enc_out_dim"]
#
#         # Transformer decoder architecture
#         self.trans_dec_in_dim = model_opts.get("dec_in_emb_dim", self.trans_enc_out_dim)
#         self.trans_dec_out_dim = model_opts["dec_out_dim"]
#         self.output_dim = model_opts["output_dim"]
#
#         self.src_mask = None
#
#         # Build encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             self.trans_enc_in_dim,
#             self.num_heads,
#             dim_feedforward=self.trans_enc_out_dim,
#             dropout=self.dropout,
#             activation=F.mish,
#             batch_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
#
#         # Build decoder
#         decoder_layer = nn.TransformerDecoderLayer(
#             self.trans_dec_in_dim,
#             self.num_heads,
#             dim_feedforward=self.trans_dec_out_dim,
#             dropout=self.dropout,
#             activation=F.mish,
#             batch_first=True,
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
#
#         # self.transformer = nn.Transformer(
#         #     d_model=self.trans_enc_in_dim,
#         #     nhead=self.num_heads,
#         #     num_encoder_layers=self.num_layers,
#         #     num_decoder_layers=self.num_layers,
#         #     dim_feedforward=self.trans_dec_in_dim,
#         #     dropout=self.dropout,
#         #     activation=F.mish,
#         #     batch_first=True,  # bs x ts x feature
#         # )
#
#     @overload
#     def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: ...
#
#     @overload
#     def forward(self, data: T_intentBatch) -> torch.Tensor: ...
#
#     @override
#     def forward(
#         self, data: T_intentBatch | tuple[torch.Tensor, torch.Tensor]
#     ) -> torch.Tensor:
#         # raise NotImplementedError
#         # TODO: (chris) Finish writing up this method.
#
#         enc_input: torch.Tensor
#         tgt_input: torch.Tensor
#         match data:
#             case dict():
#                 enc_input = data["bboxes"][:, : self.args.observe_length, :].type(
#                     FloatTensor
#                 )
#                 tgt_input = data["bboxes"][:, 1 : self.args.observe_length + 1, :].type(
#                     FloatTensor
#                 )
#             case _:
#                 enc_input = data[0]
#                 tgt_input = data[1]
#
#         ts: int
#         if enc_input.dim() == 3:
#             _, ts, _ = enc_input.shape
#         else:
#             ts, _ = enc_input.shape
#
#         mask_input: torch.Tensor = Transformer.generate_square_subsequent_mask(
#             ts, DEVICE, FloatTensor
#         )
#         mask_target: torch.Tensor = Transformer.generate_square_subsequent_mask(
#             ts, DEVICE, FloatTensor
#         )
#
#         memory: torch.Tensor = self.encoder(enc_input, mask=mask_input, is_causal=True)
#
#         output: torch.Tensor = self.decoder(
#             tgt_input, memory=memory, tgt_mask=mask_target, tgt_is_causal=True
#         )
#         if has_mask:
#             device = src.device
#             if self.src_mask is None or self.src_mask.size(0) != len(src):
#                 mask = self._generate_square_subsequent_mask(len(src)).to(device)
#                 self.src_mask = mask
#         else:
#             self.src_mask = None
#
#         # y = self.transformer(
#         #     src=src,
#         # )


class TransformerTrajBbox(nn.Module):
    def __init__(self, args: DefaultArguments, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.args = args
        self.model_opts: ModelOpts = args.model_configs
        self.config = config
        self.transformer = TimeSeriesTransformerForPrediction(self.config)

    @overload
    def forward(
        self, x: T_intentBatch
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @overload
    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @override
    def forward(
        self, x: T_intentBatch | tuple[torch.Tensor, torch.Tensor]
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]:
        # Inputs to the transformer
        past_values: torch.Tensor
        past_time_features: torch.Tensor
        future_values: torch.Tensor
        future_time_features: torch.Tensor
        match x:
            case dict():  # type: ignore[reportUnnecessaryCondition]
                past_values = x["bboxes"][:, : self.args.observe_length, :].type(
                    FloatTensor
                )
                past_time_features = x["total_frames"][
                    :, : self.args.observe_length
                ].type(FloatTensor)
                future_values = x["bboxes"][:, self.args.observe_length :, :].type(
                    FloatTensor
                )
                future_time_features = x["total_frames"][
                    :, self.args.observe_length :
                ].type(FloatTensor)
            case tuple():
                past_values = x[0][:, : self.args.observe_length, :].type(FloatTensor)
                future_values = x[0][:, self.args.observe_length :, :].type(FloatTensor)
                past_time_features = x[1][:, : self.args.observe_length].type(
                    FloatTensor
                )
                future_time_features = x[1][:, self.args.observe_length :].type(
                    FloatTensor
                )
            case _:
                past_values = x[:, : self.args.observe_length, :].type(FloatTensor)
                future_values = x[:, self.args.observe_length :, :].type(FloatTensor)
                past_time_features = x[:, : self.args.observe_length].type(FloatTensor)
                future_time_features = x[:, self.args.observe_length :].type(
                    FloatTensor
                )

        past_time_features = past_time_features.unsqueeze(2)
        future_time_features = future_time_features.unsqueeze(2)
        bs, ts, _ = past_values.shape
        past_observed_mask: torch.Tensor = torch.ones_like(
            past_values, dtype=torch.bool
        )
        outputs: Seq2SeqTSPredictionOutput = self.transformer(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
            future_time_features=future_time_features,
        )

        return outputs

    @overload
    def generate(self, x: T_intentBatch) -> torch.Tensor: ...

    @overload
    def generate(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: ...

    def generate(
        self, x: T_intentBatch | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        past_values: torch.Tensor
        past_time_features: torch.Tensor
        future_time_features: torch.Tensor
        match x:
            case dict():  # type: ignore[reportUnnecessaryCondition]
                past_values = x["bboxes"][:, : self.args.observe_length, :].type(
                    FloatTensor
                )
                past_time_features = x["total_frames"][
                    :, : self.args.observe_length
                ].type(FloatTensor)
                future_time_features = x["total_frames"][
                    :, self.args.observe_length :
                ].type(FloatTensor)
            case tuple():
                past_values = x[0][:, : self.args.observe_length, :].type(FloatTensor)
                past_time_features = x[1][:, : self.args.observe_length].type(
                    FloatTensor
                )
                future_time_features = x[1][:, self.args.observe_length :].type(
                    FloatTensor
                )
            case _:
                past_values = x[:, : self.args.observe_length, :].type(FloatTensor)
                past_time_features = x[:, : self.args.observe_length].type(FloatTensor)
                future_time_features = x[:, self.args.observe_length :].type(
                    FloatTensor
                )

        past_time_features = past_time_features.unsqueeze(2)
        future_time_features = future_time_features.unsqueeze(2)
        bs, ts, _ = past_values.shape
        past_observed_mask: torch.Tensor = torch.ones_like(
            past_values, dtype=torch.bool
        )
        outputs: SampleTSPredictionOutput = self.transformer.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
        )

        preds = outputs.sequences.mean(dim=1)
        return preds

    def build_optimizer(self, args: DefaultArguments) -> tuple[Optimizer, LRScheduler]:
        learning_rate = args.lr

        opt_eps = self.model_opts.get("opt_eps", 1e-4)
        opt_wd = self.model_opts.get("opt_wd", 1e-2)
        opt_name = self.model_opts.get("optimizer", "Adam")
        opt_mom = self.model_opts.get("momentum", 0.9)

        optimizer: Optimizer
        params = self.transformer.parameters()
        match opt_name:
            case "AdamW":
                optimizer = AdamW(
                    params,
                    lr=learning_rate,
                    eps=opt_eps,
                    weight_decay=opt_wd,
                    fused=CUDA,
                )

            case "SGD":
                optimizer = SGD(params, lr=learning_rate, momentum=opt_mom, fused=CUDA)

            case _:
                optimizer = Adam(
                    params,
                    lr=learning_rate,
                    eps=opt_eps,
                    weight_decay=opt_wd,
                    fused=CUDA,
                )

        for opt_param_grp in optimizer.param_groups:
            opt_param_grp["lr0"] = opt_param_grp["lr"]

        sched_name = self.model_opts.get("scheduler", "ExponentialLR")
        scheduler: LRScheduler
        match sched_name:
            case "OneCycleLR":
                sched_div_factor = self.model_opts.get("scheduler_div_factor", 10)
                sched_pct_start = self.model_opts.get("scheduler_pct_start", 0.2)

                scheduler = OneCycleLR(
                    optimizer,
                    learning_rate,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    div_factor=sched_div_factor,
                    pct_start=sched_pct_start,
                )

            case "ReduceLROnPlateau":
                sched_factor = self.model_opts.get("scheduler_factor", 10**0.5 * 0.1)
                sched_threshold = self.model_opts.get("scheduler_threshold", 1e-2)

                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=sched_factor,
                    threshold=sched_threshold,
                )

            case _:
                sched_gamma = self.model_opts.get("scheduler_gamma", 0.9)
                scheduler = ExponentialLR(optimizer, gamma=sched_gamma)

        return optimizer, scheduler
