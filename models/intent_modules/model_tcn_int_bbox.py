import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from models.TCN.tcn import TemporalConvNet

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class TCNINTBbox(nn.Module):
    def __init__(self,args, model_configs):
        super(TCNINTBbox, self).__init__()
        self.args = args
        self.model_configs = model_configs
        self.observe_length = self.args.observe_length
        self.predict_length = self.args.predict_length

        self.backbone = args.backbone
        self.intent_predictor = TCNINT(self.args, self.model_configs['intent_model_opts'])

        self.module_list = self.intent_predictor.module_list
        self.network_list = [self.intent_predictor]

    def forward(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        assert bbox.shape[1] == self.observe_length

        intent_pred = self.intent_predictor(bbox)
        return intent_pred.squeeze()

    def build_optimizer(self, args):
        param_group = []
        learning_rate = args.lr
        if self.backbone is not None:
            for name, param in self.backbone.named_parameters():
                if not self.args.freeze_backbone:
                    param.requires_grad = True
                    param_group +=[{'params':param, 'lr':learning_rate * 0.1}]
                else:
                    param.requires_grad = False

        for net in self.network_list:
            for module in net.module_list:
                param_group += [{'params': module.parameters(), 'lr':learning_rate}]

        optimizer = torch.optim.Adam(param_group, lr = args.lr,eps=1e-7)

        for param_group in optimizer.param_groups:
            param_group['lr0'] = param_group['lr']

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, scheduler

    def lr_scheduler(self, cur_epoch, args, gamma = 10, power = 0.75):
        decay = (1 + gamma * cur_epoch / args.epochs) ** (-power)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def predict_intent(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        assert bbox.shape[1] == self.observe_length

        intent_pred = self.intent_predictor(bbox)
        return intent_pred.squeeze()

class TCNINT(nn.Module):
    def __init__(self, args, model_opts):
        super(TCNINT, self).__init__()

        self.enc_in_dim = model_opts['enc_in_dim']
        self.enc_out_dim = model_opts['enc_out_dim']  # Assign as attribute
        self.output_dim = model_opts['output_dim']
        self.n_layers = model_opts['n_layers']
        self.kernel_size = model_opts['kernel_size']
        self.dropout = model_opts['dropout']
        self.use_skip_connections = model_opts.get('use_skip_connections', False)

        self.args = args

        self.tcn = TemporalConvNet(num_inputs=self.enc_in_dim,
                                   num_channels=[self.enc_out_dim] * model_opts['n_layers'],
                                   kernel_size=self.kernel_size, dropout=self.dropout,
                                   use_skip_connections=self.use_skip_connections)

        self.fc = nn.Sequential(
            nn.Linear(self.enc_out_dim, 16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(16, self.output_dim)
        )

        if model_opts['output_activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif model_opts['output_activation'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.module_list = [self.tcn, self.fc]

    def forward(self, enc_input):
        tcn_output = self.tcn(enc_input.transpose(1, 2)).transpose(1, 2)
        tcn_last_output = tcn_output[:, -1:, :]
        output = self.fc(tcn_last_output)
        outputs = output.unsqueeze(1)
        return outputs


