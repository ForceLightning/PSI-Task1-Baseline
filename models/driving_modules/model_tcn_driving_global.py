from typing import Any, overload

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
)

from data.custom_dataset import T_drivingBatch, T_drivingSample
from models.TCN.tcn import TemporalConvNet
from models.driving_modules.model_lstm_driving_global import ResCNNEncoder
from utils.args import DefaultArguments, ModelOpts
from utils.cuda import *


class ResTCNDrivingGlobal(nn.Module):
    def __init__(self, args: DefaultArguments, model_opts: ModelOpts) -> None:
        super().__init__()

        self.observe_length = args.observe_length
        self.predict_length = args.predict_length
        self.args = args
        self.output_dim = model_opts["output_dim"]
        self.model_opts = model_opts

        # CNN Encoder
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        dropout_p = 0.0  # dropout probability

        # TCN Decoder Architecture
        self.TCN_enc_in_dim = model_opts["enc_in_dim"]
        self.TCN_dec_out_dim = model_opts["dec_out_dim"]
        self.TCN_hidden_layers = model_opts["n_layers"]
        self.TCN_dropout = model_opts["dropout"]
        self.TCN_kernel_size = model_opts["kernel_size"]
        self.TCN_skip_connections = model_opts.get("use_skip_connections", False)

        self.cnn_encoder = ResCNNEncoder(
            CNN_fc_hidden1, CNN_fc_hidden2, dropout_p, CNN_embed_dim, args
        )

        self.tcn_speed = TemporalConvNet(
            num_inputs=CNN_embed_dim + self.TCN_enc_in_dim,
            num_channels=[self.TCN_dec_out_dim] * self.TCN_hidden_layers,
            kernel_size=self.TCN_kernel_size,
            dropout=self.TCN_dropout,
            use_skip_connections=self.TCN_skip_connections,
        )

        self.tcn_dir = TemporalConvNet(
            num_inputs=CNN_embed_dim + self.TCN_enc_in_dim,
            num_channels=[self.TCN_dec_out_dim] * self.TCN_hidden_layers,
            kernel_size=self.TCN_kernel_size,
            dropout=self.TCN_dropout,
            use_skip_connections=self.TCN_skip_connections,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.TCN_dec_out_dim * self.observe_length, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.Mish(),
            nn.Dropout(self.TCN_dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.Mish(),
            nn.Dropout(self.TCN_dropout),
            nn.Linear(64, self.output_dim * self.predict_length),
        )

        match model_opts["output_activation"]:
            case "tanh":
                self.activation = nn.Tanh()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case _:
                self.activation = nn.Identity()

        self.module_list = [self.cnn_encoder, self.tcn_speed, self.tcn_dir, self.fc]

        self.intent_embedding = "int" in self.args.model_name
        self.reason_embedding = "rsn" in self.args.model_name
        self.speed_embedding = "speed" in self.args.model_name

    @overload
    def forward(self, data: torch.Tensor) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        data: T_drivingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        data: T_drivingBatch | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        visual_feats: torch.Tensor
        match data:
            case dict():  # The TypedDict becomes a plain dict at runtime.
                # Handle loading embeddings from file (already loaded into the dataset instance)
                if self.args.freeze_backbone:
                    visuals = data["global_featmaps"]
                else:
                    visuals = data["image"][
                        :, : self.args.observe_length, :, :, :
                    ].type(FloatTensor)
                assert (
                    visuals.shape[1] == self.args.observe_length
                ), f"Temporal dimension of visual embeddings or images {visuals.shape[1]} does not match `observe_length` {self.args.observe_length}"

                visual_feats = self.cnn_encoder(visuals)
            case _:
                assert (
                    data.shape[1] == self.args.observe_length
                ), f"Temporal dimension of visual embeddings or images {data.shape[1]} does not match `observe_length` {self.args.observe_length}"
                visual_feats = self.cnn_encoder(data)

        tcn_input = visual_feats  # bs x (512 + 4) x ts

        tcn_speed_output: torch.Tensor = self.tcn_speed(
            tcn_input.transpose(1, 2)
        )  # bs x tcn_dec_out_dim x ts
        tcn_speed_output = tcn_speed_output.reshape(
            -1, self.TCN_dec_out_dim * self.observe_length
        )
        # tcn_last_speed_output = tcn_speed_output[:, -1:, :]

        tcn_dir_output: torch.Tensor = self.tcn_dir(tcn_input.transpose(1, 2))
        tcn_dir_output = tcn_dir_output.reshape(
            -1, self.TCN_dec_out_dim * self.observe_length
        )
        # tcn_last_dir_output = tcn_dir_output[:, -1:, :]

        speed_output: torch.Tensor = self.fc(tcn_speed_output)
        dir_output: torch.Tensor = self.fc(tcn_dir_output)

        # output = self.fc(tcn_speed_output)
        # output = self.activation(output).reshape(
        #     -1, self.predict_length, self.output_dim
        # )
        if isinstance(data, torch.Tensor):
            return torch.cat([speed_output, dir_output], dim=0)
        return speed_output, dir_output

    def build_optimizer(self, args: DefaultArguments) -> tuple[Optimizer, LRScheduler]:
        param_group: list[dict[str, Any]] = []
        learning_rate = args.lr

        for module in self.module_list:
            match module:
                case ResCNNEncoder():
                    if self.args.freeze_backbone:
                        submodules = [
                            module.fc1,
                            module.bn1,
                            module.fc2,
                            module.bn2,
                            module.fc3,
                        ]
                    else:
                        submodules = [
                            module.resnet,
                            module.fc1,
                            module.bn1,
                            module.fc2,
                            module.bn2,
                            module.fc3,
                        ]
                    for submodule in submodules:
                        param_group += [
                            {"params": submodule.parameters(), "lr": learning_rate}
                        ]
                case _:
                    param_group += [
                        {"params": module.parameters(), "lr": learning_rate}
                    ]

        opt_eps = self.model_opts.get("opt_eps", 1e-4)
        opt_wd = self.model_opts.get("opt_wd", 1e-2)
        opt_name = self.model_opts.get("optimizer", "Adam")
        opt_mom = self.model_opts.get("momentum", 0.9)

        optimizer: Optimizer
        match opt_name:
            case "AdamW":
                optimizer = AdamW(
                    param_group,
                    lr=learning_rate,
                    eps=opt_eps,
                    weight_decay=opt_wd,
                    fused=CUDA,
                )
            case "SGD":
                optimizer = SGD(
                    param_group, lr=learning_rate, momentum=opt_mom, fused=CUDA
                )
            case _:
                optimizer = Adam(
                    param_group,
                    lr=learning_rate,
                    eps=opt_eps,
                    weight_decay=opt_wd,
                    fused=CUDA,
                )

        for opt_param_group in optimizer.param_groups:
            opt_param_group["lr0"] = opt_param_group["lr"]

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
