import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from models.TCAN.custom_transformer import MultiheadAttention
from models.TCN.tcn import Chomp1d
from utils.cuda import *


class AttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        key_size: int,
        value_size: int,
        num_heads: int,
        seq_length: int,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.attention = MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            kdim=key_size,
            vdim=value_size,
            batch_first=True,
        )
        # Causal attention mask.
        # WARNING: This actually causes NaN values to show up due to the implementation of the
        # Softmax function. See: https://github.com/pytorch/pytorch/issues/41508
        self.attn_mask = torch.triu(
            torch.ones((self.seq_length, self.seq_length), dtype=torch.bool), diagonal=1
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_output_weights = self.attention(
            x,
            x,
            x,
            attn_mask=self.attn_mask,
            # is_causal=True,
        )
        return attn_output, attn_output_weights


class TemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        key_size: int,
        num_sub_blocks: int,
        temp_attn: bool,
        num_heads: int,
        en_res: bool,
        conv: bool,
        stride: int,
        dilation: int,
        padding: int,
        visual: bool,
        seq_length: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.visual = visual
        self.en_res = en_res
        self.conv = conv
        self.temp_attn = temp_attn
        self.seq_length = seq_length
        if self.temp_attn:
            if self.num_heads > 1:
                self.attentions = AttentionBlock(
                    n_inputs, key_size, n_inputs, num_heads, seq_length
                )
                self.add_module("attention", self.attentions)
                self.linear_cat = nn.Linear(n_inputs, n_inputs)
            else:
                self.attention = AttentionBlock(
                    n_inputs, key_size, n_inputs, 1, seq_length
                )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        if self.conv:
            self.net = self._make_layers(
                num_sub_blocks,
                n_inputs,
                n_outputs,
                kernel_size,
                stride,
                dilation,
                padding,
                dropout,
            )
            self.init_weights()

    def _make_layers(
        self,
        num_sub_blocks: int,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ) -> nn.Module:
        layers_list = [
            weight_norm(
                nn.Conv1d(n_inputs, n_outputs, kernel_size, stride, padding, dilation)
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        for _ in range(num_sub_blocks - 1):
            layers_list += [
                weight_norm(
                    nn.Conv1d(
                        n_outputs, n_outputs, kernel_size, stride, padding, dilation
                    )
                ),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        return nn.Sequential(*layers_list)

    def init_weights(self):
        layer_idx_list: list[int] = []
        for name, _ in self.net.named_parameters():
            inlayer_param_list = name.split(".")
            layer_idx_list.append(int(inlayer_param_list[0]))
        layer_idxes = list(set(layer_idx_list))
        for idx in layer_idxes:
            getattr(self.net[idx], "weight").data.normal_(0, 0.01)

        if self.downsample is not None:
            _ = self.downsample.weight.data.normal_(0, 0.01)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.temp_attn:
            en_res_x = None
            if self.num_heads > 1:
                # num_heads x bs x ts x emb_size
                x_out, attn_weight = self.attentions(x)
                # out = self.net(self.linear_cat(x_out.transpose(1, 2)).transpose(1, 2))
                linear_out = self.linear_cat(x_out)
                out = self.net(linear_out.transpose(1, 2)).transpose(1, 2)
            else:
                out_attn, attn_weight = self.attention(x)
                if self.conv:
                    out = self.net(out_attn.transpose(1, 2)).transpose(1, 2)
                else:
                    out = out_attn
                weight_x = F.softmax(attn_weight.sum(dim=2), dim=1)
                en_res_x = (
                    weight_x.unsqueeze(2).repeat(1, 1, x.size(1)).transpose(1, 2) * x
                )
                en_res_x = (
                    en_res_x if self.downsample is None else self.downsample(en_res_x)
                )

            res = (
                x
                if self.downsample is None
                else self.downsample(x.transpose(1, 2)).transpose(1, 2)
            )

            if self.visual:
                attn_weight_cpu = attn_weight.detach().cpu()
            else:
                attn_weight_cpu = torch.zeros_like(attn_weight)
            del attn_weight

            if self.en_res and en_res_x is not None:
                return self.relu(out + res + en_res_x), attn_weight_cpu
            else:
                return self.relu(out + res), attn_weight_cpu
        else:
            out: torch.Tensor = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)  # bs x emb_size x T


class TemporalConvAttnNet(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_channels: list[int],
        num_sub_blocks: int,
        temp_attn: bool,
        num_heads: int,
        en_res: bool,
        conv: bool,
        key_size: int,
        kernel_size: int,
        visual: bool,
        seq_length: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[TemporalAttentionBlock] = []
        self.temp_attn = temp_attn
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size: int = 2**i
            in_channels = emb_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalAttentionBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    in_channels,
                    num_sub_blocks,
                    temp_attn,
                    num_heads,
                    en_res,
                    conv,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    visual=visual,
                    seq_length=seq_length,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]] | torch.Tensor:
        attn_weight_list: list[list[torch.Tensor]] = []
        if self.temp_attn:
            out = x
            for i in range(len(self.network)):
                out, attn_weight = self.network[i](out)
                attn_weight_list.append([attn_weight[0], attn_weight[-1]])
            return out, attn_weight_list
        else:
            return self.network(x)
