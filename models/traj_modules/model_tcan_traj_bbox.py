from typing import Any

import numpy as np
import torch
from torch import nn

from models.TCAN.tcan import TemporalConvAttnNet
from utils.args import DefaultArguments, ModelOpts
from utils.cuda import *


class TCANTrajBbox(nn.Module):
    def __init__(self, args: DefaultArguments, model_opts: ModelOpts) -> None:
        super().__init__()

        self.args = args
        self.observe_length = args.observe_length
        self.predict_length = args.predict_length
        self.output_dim = model_opts["output_dim"]

        # TCAN Architecture
        self.TCAN_enc_in_dim = model_opts["enc_in_dim"]
        self.TCAN_dec_out_dim = model_opts["dec_out_dim"]
        self.TCAN_hidden_layers = model_opts["n_layers"]
        self.TCAN_dropout = model_opts["dropout"]
        self.TCAN_kernel_size = model_opts["kernel_size"]
        self.TCAN_skip_connections = model_opts.get("use_skip_connections", False)
        self.TCAN_num_heads = model_opts.get("num_heads", 4)
        self.temp_attn = True

        self.tcan = TemporalConvAttnNet(
            emb_size=self.TCAN_enc_in_dim,
            num_channels=[self.TCAN_dec_out_dim] * self.TCAN_hidden_layers,
            num_sub_blocks=2,
            temp_attn=self.temp_attn,
            num_heads=self.TCAN_num_heads,
            en_res=True,
            conv=True,
            key_size=self.TCAN_enc_in_dim,
            kernel_size=self.TCAN_kernel_size,
            visual=True,
            seq_length=self.observe_length,
            dropout=self.TCAN_dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.TCAN_dec_out_dim * self.observe_length, 16),
            nn.Mish(),
            nn.Dropout(self.TCAN_dropout),
            nn.Linear(16, self.output_dim * self.predict_length),
        )

        match model_opts["output_activation"]:
            case "tanh":
                self.activation = nn.Tanh()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case _:
                self.activation = nn.Identity()

        self.module_list = [self.tcan, self.fc]
        self.intent_reasoning = "int" in self.args.model_name
        self.reason_embedding = "rsn" in self.args.model_name
        self.speed_embedding = "speed" in self.args.model_name

    def forward(
        self,
        data: (
            dict[str, torch.Tensor | np.ndarray[Any, Any] | float | int] | torch.Tensor
        ),
    ) -> torch.Tensor:
        if isinstance(data, dict):
            bbox = data["bboxes"][:, : self.args.observe_length, :].type(FloatTensor)
        else:
            bbox = data
        assert (
            bbox.shape[1] == self.args.observe_length
        ), "bbox temporal dimension size does not match `observe_length`"
        enc_input = bbox

        tcan_output: torch.Tensor
        if self.temp_attn:
            tcan_output, _ = self.tcan(enc_input)
        else:
            tcan_output = self.tcan(enc_input)

        tcan_output = tcan_output.transpose(1, 2)
        tcan_output = tcan_output.reshape(
            -1, self.TCAN_dec_out_dim * self.observe_length
        )

        output: torch.Tensor = self.fc(tcan_output)
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )
        return output

    def build_optimizer(
        self, args: DefaultArguments
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        param_group: list[dict[str, Any]] = []
        learning_rate = args.lr

        for module in self.module_list:
            for _, param in module.named_parameters():
                param.requires_grad = True
                param_group += [{"params": param, "lr": learning_rate}]

        optimizer = torch.optim.Adam(
            param_group, lr=args.lr, eps=1e-7, foreach=False, fused=True
        )

        for opt_param_group in optimizer.param_groups:
            opt_param_group["lr0"] = opt_param_group["lr"]

        # WARNING: Breaking change: Optimizer to use a one cycle learning rate policy instead.

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     learning_rate,
        #     epochs=args.epochs,
        #     steps_per_epoch=args.steps_per_epoch,
        #     div_factor=10,
        #     pct_start=0.2,
        # )

        return optimizer, scheduler


class TCANTrajBboxInt(TCANTrajBbox):
    def forward(
        self, data: dict[str, torch.Tensor | np.ndarray[Any, Any] | float | int]
    ) -> torch.Tensor:

        # bs x ts x 4
        bbox: torch.Tensor = data["bboxes"][:, : self.args.observe_length, :].type(
            FloatTensor
        )

        intent: torch.Tensor = data["intention_prob"][
            :, : self.args.observe_length
        ].type(FloatTensor)
        # bs x ts x 1
        intent = intent.unsqueeze(2)
        assert (
            bbox.shape[1] == self.args.observe_length
        ), "bbox temporal dimension does not match `observe_length`"
        assert (
            intent.shape[1] == self.args.observe_length
        ), "intent temporal dimension does not match `observe_length`"

        # bs x ts x 5
        enc_input = torch.cat([bbox, intent], dim=2)

        tcan_output: torch.Tensor
        if self.temp_attn:
            tcan_output, _ = self.tcan(enc_input)
        else:
            tcan_output = self.tcan(enc_input)

        tcan_output = tcan_output.transpose(1, 2)
        tcan_output = tcan_output.reshape(
            -1, self.TCAN_dec_out_dim * self.observe_length
        )

        output: torch.Tensor = self.fc(tcan_output)
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )
        return output
