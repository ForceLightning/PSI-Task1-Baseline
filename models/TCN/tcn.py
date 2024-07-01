from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from utils.cuda import *


class Chomp1d(nn.Module):
    """Removes the last `chomp_size` elements from the input tensor.

    :param int chomp_size: Number of elements to remove from the input tensor.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A temporal block that consists of dilated causal convolutions with a residual connection.

    :param int n_inputs: Number of input channels.
    :param int n_outputs: Number of output channels.
    :param int kernel_size: Size of the convolutional kernel.
    :param int stride: Stride of the convolutional kernel.
    :param int dilation: Dilation factor of the convolutional kernel.
    :param int padding: Padding of the convolutional kernel.
    :param float dropout: Dropout rate.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the convolutional layers with a normal
        distribution.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalSkipBlock(TemporalBlock):
    """A temporal block with a residual connection and skip connections.

    :param int n_inputs: Number of input channels.
    :param int n_outputs: Number of output channels.
    :param int kernel_size: Size of the convolutional kernel.
    :param int stride: Stride of the convolutional kernel.
    :param int dilation: Dilation factor of the convolutional kernel.
    :param int padding: Padding of the convolutional kernel.
    :param float dropout: Dropout rate.
    :param bool use_skips: Whether to use skip connections.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        use_skips: bool = False,
    ) -> None:
        super().__init__(
            n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=dropout
        )
        self.use_skips = use_skips
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)

    @override
    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_skips:
            out: torch.Tensor
            res: torch.Tensor
            if type(x) == tuple:
                x, skip = x[0], x[1]
                out = self.net(x)
                res = self.downsample(x)
                return (self.relu(torch.add(out, res)), torch.add(skip, res.clone()))
            else:
                out = self.net(x)
                res = self.downsample(x)
                return (self.relu(torch.add(out, res)), res)
        else:
            out = self.net(x)
            res = self.downsample(x)
            return self.relu(torch.add(out, res))


class TemporalConvNet(nn.Module):
    """A Temporal Convolutional Network (TCN) that consists of multiple temporal blocks.

    :param int num_inputs: Number of input channels.
    :param num_channels: List of the number of channels for each temporal block.
    :type num_channels: list[int]
    :param int kernel_size: Size of the convolutional kernel.
    :param float dropout: Dropout rate.
    :param bool use_skip_connections: Whether to use skip connections.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        layers: list[TemporalSkipBlock | TemporalBlock] = []

        # Create temporal convolutional layers based on the number of channels
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size: int = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Use TemporalSkipBlock if use_skip_connections is True
            if use_skip_connections:
                layers += [
                    TemporalSkipBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size,
                        dropout=dropout,
                        use_skips=True,
                    )
                ]
            else:
                layers += [
                    TemporalBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size,
                        dropout=dropout,
                    )
                ]

        self.network = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
