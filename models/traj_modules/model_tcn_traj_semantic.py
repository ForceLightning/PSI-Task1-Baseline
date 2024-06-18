from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torchvision import models

from models.TCN.tcn import TemporalConvAttnNet, TemporalConvNet
from models.driving_modules.model_lstm_driving_global import ResCNNEncoder
from utils.args import DefaultArguments
from utils.cuda import *

# TODO(chris): Rename file to model_tcn_traj_global.py


class TCNTrajGlobal(nn.Module):
    def __init__(
        self, args: DefaultArguments, model_opts: dict[str, int | float]
    ) -> None:
        super().__init__()

        self.observe_length = args.observe_length
        self.predict_length = args.predict_length
        self.args = args
        self.predict_length = model_opts["predict_length"]
        self.output_dim = model_opts["output_dim"]

        # EncoderCNN architecture
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        res_size = 224  # ResNet image size
        dropout_p = 0.0  # dropout probability

        # DecoderTCN architecture
        self.TCN_enc_in_dim = model_opts["enc_in_dim"]
        self.TCN_dec_out_dim = model_opts["dec_out_dim"]
        self.TCN_hidden_layers = model_opts["n_layers"]
        self.TCN_dropout = model_opts["dropout"]
        self.TCN_kernel_size = model_opts["kernel_size"]
        self.TCN_skip_connections = model_opts.get("use_skip_connections", False)

        self.cnn_encoder = ResCNNEncoder(
            CNN_fc_hidden1, CNN_fc_hidden2, dropout_p, CNN_embed_dim, args
        )
        self.tcn = TemporalConvNet(
            num_inputs=CNN_embed_dim + self.TCN_enc_in_dim,
            num_channels=[self.TCN_dec_out_dim] * self.TCN_hidden_layers,
            kernel_size=self.TCN_kernel_size,
            dropout=self.TCN_dropout,
            use_skip_connections=self.TCN_skip_connections,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.TCN_dec_out_dim * self.observe_length, 64),
            nn.BatchNorm1d(64, momentum=0.01),
            nn.Mish(),
            nn.Dropout(self.TCN_dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=0.01),
            nn.Mish(),
            nn.Dropout(self.TCN_dropout),
            nn.Linear(32, self.output_dim * self.predict_length),
        )

        match model_opts["output_activation"]:
            case "tanh":
                self.activation = nn.Tanh()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case _:
                self.activation = nn.Identity()

        self.module_list = [self.cnn_encoder, self.tcn, self.fc]

        self.intent_embedding = "int" in self.args.model_name
        self.reason_embedding = "rsn" in self.args.model_name
        self.speed_embedding = "speed" in self.args.model_name

    def forward(self, data):
        visual_feats = None

        # Handle loading embeddings from file (already loaded into the dataset instance)
        if self.args.freeze_backbone:
            visual_embeddings = data["global_featmaps"]
            visual_feats = self.cnn_encoder(visual_embeddings)
        else:
            images = data["image"][:, : self.args.observe_length, :, :, :].type(
                FloatTensor
            )
            assert images.shape[1] == self.args.observe_length
            visual_feats = self.cnn_encoder(images)  # bs x ts x 256
        bbox = data["bboxes"][:, : self.args.observe_length, :].type(
            FloatTensor
        )  # bs x ts x 4

        assert bbox.shape[1] == self.args.observe_length

        tcn_input = torch.cat([visual_feats, bbox], dim=2)  # bs x (512 + 4) x ts

        tcn_output = self.tcn(tcn_input.transpose(1, 2))  # bs x tcn_dec_out_dim x ts
        tcn_output = tcn_output.reshape(-1, self.TCN_dec_out_dim * self.observe_length)
        # tcn_last_output = tcn_output[:, -1:, :]
        # output = self.fc(tcn_last_output)
        output = self.fc(tcn_output)
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )
        return output

    def build_optimizer(self, args):
        param_group = []
        learning_rate = args.lr

        for module in self.module_list:
            # Only use the non-pretrained layers of the CNN.
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

        optimizer = torch.optim.Adam(param_group, lr=learning_rate, eps=1e-7)

        for param_group in optimizer.param_groups:
            param_group["lr0"] = param_group["lr"]

        # ! Breaking change: optimizer to use a one cycle learning rate policy instead.
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     learning_rate,
        #     epochs=args.epochs,
        #     steps_per_epoch=args.steps_per_epoch,
        #     div_factor=10,
        # )

        return optimizer, scheduler

    def lr_scheduler(self, optimizer, cur_epoch, args, gamma=10, power=0.75) -> None:
        decay = (1 + gamma * cur_epoch / args.epoch) ** (-power)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr0"] * decay
            param_group["weight_decay"] = 1e-3
            param_group["momentum"] = 0.9
            param_group["nesterov"] = True
        return

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)


class TCANTrajGlobal(TCNTrajGlobal):
    def __init__(
        self, args: DefaultArguments, model_opts: dict[str, float | int]
    ) -> None:
        super().__init__(args, model_opts)
        self.TCN_num_heads = model_opts["num_heads"]
        self.temp_attn = True
        self.tcn = TemporalConvAttnNet(
            emb_size=self.TCN_enc_in_dim,
            num_channels=[self.TCN_dec_out_dim] * self.TCN_hidden_layers,
            num_sub_blocks=2,
            temp_attn=self.temp_attn,
            num_heads=self.TCN_num_heads,
            en_res=True,
            conv=True,
            key_size=self.TCN_enc_in_dim,
            kernel_size=self.TCN_kernel_size,
            visual=True,
            seq_length=self.observe_length,
            dropout=self.TCN_dropout,
        )

        self.module_list = [self.cnn_encoder, self.tcn, self.fc]

    def forward(
        self, data: dict[str, torch.Tensor | np.ndarray[Any, Any] | float | int]
    ) -> tuple[torch.Tensor | list[float]] | torch.Tensor:
        # bs x ts x 256
        visual_feats = None

        # Handle loading embeddings from file (already loaded into the dataset instance)
        if self.args.freeze_backbone:
            visual_embeddings = data["global_featmaps"]
            visual_feats = self.cnn_encoder(visual_embeddings)
        else:
            images = data["image"][:, : self.args.observe_length, :, :, :].type(
                FloatTensor
            )
            assert images.shape[1] == self.args.observe_length
            visual_feats = self.cnn_encoder(images)

        # bs x ts x 4
        bbox = data["bboxes"][:, : self.args.observe_length, :].type(FloatTensor)

        assert bbox.shape[1] == self.args.observe_length

        # TCAN input should be (bs, channels, ts)
        tcn_input = torch.cat([visual_feats, bbox], dim=2)  # bs x ts x (512 + 4)
        # tcn_input = tcn_input.transpose(1, 2)  # bs x (512 + 4) x ts

        tcn_output, attn_weight_list = None, None
        if self.temp_attn:
            tcn_output, attn_weight_list = self.tcn(tcn_input)
        else:
            tcn_output = self.tcn(tcn_input)

        tcn_output = tcn_output.transpose(1, 2)
        tcn_output = tcn_output.reshape(-1, self.TCN_dec_out_dim * self.observe_length)

        output = self.fc(tcn_output)
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )

        return output
