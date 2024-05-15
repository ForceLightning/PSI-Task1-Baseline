import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalSkipBlock(TemporalBlock):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_skips=False):
        super().__init__(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=dropout)
        self.use_skips = use_skips
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)

    def init_weights(self):
        super().init_weights()

    def forward(self, x):
        if self.use_skips:
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, use_skip_connections=False):
        super(TemporalConvNet, self).__init__()
        layers = []

        # Create temporal convolutional layers based on the number of channels
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Use TemporalSkipBlock if use_skip_connections is True
            if use_skip_connections:
                layers += [TemporalSkipBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                             padding=(kernel_size - 1) * dilation_size, dropout=dropout,
                                             use_skips=True)]
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)