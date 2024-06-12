import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torchvision import models

from models.TCN.tcn import TemporalConvNet
from models.driving_modules.model_lstm_driving_global import ResCNNEncoder

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# TODO(chris): Rename file to model_tcn_traj_global.py

class TCNTrajGlobal(nn.Module):
    def __init__(self, args, model_opts) -> None:
        super().__init__()

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
        TCN_enc_in_dim = model_opts["enc_in_dim"]
        TCN_dec_out_dim = model_opts["dec_out_dim"]
        TCN_hidden_layers = model_opts["n_layers"]
        TCN_dropout = model_opts["dropout"]
        TCN_kernel_size = model_opts["kernel_size"]
        TCN_skip_connections = model_opts.get("use_skip_connections", False)

        self.cnn_encoder = ResCNNEncoder(
            CNN_fc_hidden1, CNN_fc_hidden2, dropout_p, CNN_embed_dim
        )
        self.tcn = TemporalConvNet(
            num_inputs=CNN_embed_dim + TCN_enc_in_dim,
            num_channels=[TCN_dec_out_dim] * TCN_hidden_layers,
            kernel_size=TCN_kernel_size,
            dropout=TCN_dropout,
            use_skip_connections=TCN_skip_connections,
        )

        self.fc = nn.Sequential(
            nn.Linear(TCN_dec_out_dim, 16),
            nn.Mish(),
            nn.Dropout(TCN_dropout),
            nn.Linear(16, self.output_dim * self.predict_length),
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
        images = data["image"][:, : self.args.observe_length, :, :, :].type(FloatTensor)
        bbox = data["bboxes"][:, : self.args.observe_length, :].type(
            FloatTensor
        )  # bs x ts x 4

        assert images.shape[1] == self.args.observe_length
        assert bbox.shape[1] == self.args.observe_length

        visual_feats = self.cnn_encoder(images)  # bs x ts x 256
        tcn_input = torch.cat([visual_feats, bbox], dim=2)

        tcn_output = self.tcn(tcn_input.transpose(1, 2)).transpose(1, 2)
        tcn_last_output = tcn_output[:, -1:, :]
        output = self.fc(tcn_last_output)
        output = self.activation(output).reshape(
            -1, self.predict_length, self.output_dim
        )
        return output

    def build_optimizer(self, args):
        param_group = []
        learning_rate = args.lr

        for module in self.module_list:
            # TODO: Only use the non-pretrained layers of the CNN.
            match module:
                case ResCNNEncoder():
                    for submodule in [
                        module.resnet,
                        module.fc1,
                        module.bn1,
                        module.fc2,
                        module.bn2,
                        module.fc3,
                    ]:
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

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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
