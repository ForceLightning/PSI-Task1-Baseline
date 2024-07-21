from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing_extensions import overload, override

from data.custom_dataset import T_intentBatch
from models.model_interfaces import IConstructOptimizer
from models.TCN.tcn import TemporalConvNet
from utils.args import DefaultArguments, ModelOpts
from utils.cuda import *


class TCNTrajBbox(nn.Module, IConstructOptimizer):
    """A TCN model for trajectory prediction with bbox input.

    The model consists of a Temporal Convolutional Network (TCN) that processes the
    input bbox sequence and predicts the future trajectory.

    :param DefaultArguments args: The training arguments.
    :param ModelOpts model_opts: The model options.

    :ivar int enc_in_dim: The input dimension of the encoder.
    :ivar int enc_out_dim: The output dimension of the encoder.
    :ivar int dec_in_emb_dim: The embedding dimension of the decoder input.
    :ivar int dec_out_dim: The output dimension of the decoder.
    :ivar int output_dim: The output dimension of the model.
    :ivar int n_layers: The number of layers in the TCN.
    :ivar float dropout: The dropout rate.
    :ivar int predict_length: The length of the predicted trajectory.
    :ivar int kernel_size: The kernel size of the TCN.
    :ivar bool use_skip_connections: Whether to use skip connections in the TCN.
    :ivar DefaultArguments args: The training arguments.
    :ivar nn.Module backbone: The backbone model.
    :ivar TemporalConvNet tcn: The TCN model.
    :ivar nn.Sequential fc: The fully connected layer.
    :ivar nn.Module activation: The activation function.
    :ivar list[nn.Module] module_list: The list of modules in the model.
    :ivar bool intent_embedding: Whether to use the intention embedding.
    :ivar bool reason_embedding: Whether to use the reason embedding.
    :ivar bool speed_embedding: Whether to use the speed embedding.
    """

    def __init__(self, args: DefaultArguments, model_opts: ModelOpts) -> None:
        super().__init__()
        self.enc_in_dim = model_opts["enc_in_dim"]
        self.enc_out_dim = model_opts["enc_out_dim"]
        self.dec_in_emb_dim = model_opts["dec_in_emb_dim"]
        self.dec_out_dim = model_opts["dec_out_dim"]
        self.output_dim = model_opts["output_dim"]

        self.n_layers = model_opts["n_layers"]
        self.dropout = model_opts["dropout"]
        self.predict_length = model_opts["predict_length"]
        self.kernel_size = model_opts["kernel_size"]
        self.use_skip_connections = model_opts.get("use_skip_connections", False)
        self.args = args

        self.backbone = None

        self.tcn = TemporalConvNet(
            num_inputs=self.enc_in_dim,
            num_channels=[self.enc_out_dim] * self.n_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            use_skip_connections=self.use_skip_connections,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.dec_out_dim, 16),
            nn.Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(16, self.output_dim * self.predict_length),
        )

        match model_opts["output_activation"]:
            case "tanh":
                self.activation = nn.Tanh()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case _:
                self.activation = nn.Identity()

        self.module_list = [self.tcn, self.fc]
        self.intent_embedding = "int" in self.args.model_name
        self.reason_embedding = "rsn" in self.args.model_name
        self.speed_embedding = "speed" in self.args.model_name

    @override
    def forward(self, data: T_intentBatch | torch.Tensor) -> torch.Tensor:
        enc_input: torch.Tensor
        if isinstance(data, dict):
            bbox: torch.Tensor = data["bboxes"][:, : self.args.observe_length, :].type(
                FloatTensor
            )
            enc_input = bbox
        else:
            enc_input = data

        assert enc_input.shape[1] == self.args.observe_length

        tcn_output: torch.Tensor = self.tcn(enc_input.transpose(1, 2)).transpose(1, 2)
        tcn_last_output = tcn_output[:, -1:, :]

        output: torch.Tensor = self.fc(
            tcn_last_output
        )  # bs x output_dim * predict_length
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )
        return output

    @override
    def build_optimizer(
        self, args: DefaultArguments
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        param_group: list[dict[str, Any]] = []
        learning_rate = args.lr
        if self.backbone is not None:
            for _, param in self.backbone.named_parameters():
                if not self.args.freeze_backbone:
                    param.requires_grad = True
                    param_group += [{"params": param, "lr": learning_rate * 0.1}]
                else:
                    param.requires_grad = False

        for module in self.module_list:
            param_group += [{"params": module.parameters(), "lr": learning_rate}]

        optimizer = torch.optim.Adam(
            param_group, lr=args.lr, eps=1e-7, weight_decay=1e-4
        )

        for opt_param_group in optimizer.param_groups:
            opt_param_group["lr0"] = opt_param_group["lr"]

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, scheduler

    def lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        cur_epoch: int,
        args: DefaultArguments,
        gamma: float = 10.0,
        power: float = 0.75,
    ) -> None:
        decay: float = (1 + gamma * cur_epoch / args.epochs) ** (-power)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr0"] * decay
            param_group["weight_decay"] = 1e-3
            param_group["momentum"] = 0.9
            param_group["nesterov"] = True
        return

    def _reset_parameters(self) -> None:
        _ = nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)


class TCNTrajBboxInt(TCNTrajBbox):
    """A TCN model for trajectory prediction with bbox input and intention embedding."""

    @overload
    def forward(self, data: T_intentBatch) -> torch.Tensor: ...

    @overload
    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: ...

    @override
    def forward(
        self, data: T_intentBatch | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        enc_input: torch.Tensor
        if isinstance(data, dict):
            bbox: torch.Tensor = data["bboxes"][:, : self.args.observe_length, :].type(
                FloatTensor
            )
            intent: torch.Tensor = data["intention_prob"][
                :, : self.args.observe_length
            ].type(FloatTensor)
            intent = intent.unsqueeze(2)
            assert bbox.shape[1] == self.args.observe_length
            assert intent.shape[1] == self.args.observe_length
            # enc_input = bbox
            enc_input = torch.cat([bbox, intent], dim=2)
        else:
            enc_input = torch.cat(data, dim=2)

        tcn_output: torch.Tensor = self.tcn(enc_input.transpose(1, 2)).transpose(1, 2)
        tcn_last_output = tcn_output[:, -1:, :]

        output: torch.Tensor = self.fc(
            tcn_last_output
        )  # bs x output_dim * predict_length
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )
        return output


class TCNTrajBboxPose(TCNTrajBbox):
    """A TCN model for trajectory prediction with bbox input and pose data."""

    @overload
    def forward(self, data: T_intentBatch) -> torch.Tensor: ...

    @overload
    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: ...

    @override
    def forward(
        self, data: T_intentBatch | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        enc_input: torch.Tensor
        if isinstance(data, dict):
            bbox: torch.Tensor = (
                data["bboxes"][:, : self.args.observe_length, :]
                .type(FloatTensor)
                .to(DEVICE)
            )
            poses: torch.Tensor = (
                data["pose"][:, : self.args.observe_length, :]
                .type(FloatTensor)
                .to(DEVICE)
            )
            poses = poses.reshape(-1, self.args.observe_length, 34)
            assert bbox.shape[1] == self.args.observe_length
            # enc_input = bbox
            enc_input = torch.cat([bbox, poses], dim=2)
        else:
            bbox, poses = data
            poses = poses.reshape(-1, self.args.observe_length, 34)
            enc_input = torch.cat((bbox, poses), dim=2)

        tcn_output: torch.Tensor = self.tcn(enc_input.transpose(1, 2)).transpose(1, 2)
        tcn_last_output = tcn_output[:, -1:, :]

        output: torch.Tensor = self.fc(
            tcn_last_output
        )  # bs x output_dim * predict_length
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )
        return output
