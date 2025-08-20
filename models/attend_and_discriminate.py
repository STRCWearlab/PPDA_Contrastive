import os
import torch
import torch.nn.functional as F
from torch import nn
import time

from simuclr.utils import paint
from models.base_model import BaseModel


def conv1d(
    ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = True
):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv


class SelfAttention(nn.Module):
    """
    # self-attention implementation from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    Self attention layer for nd
    """

    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels // div)
            self.key = conv1d(n_channels, n_channels // div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    """
    Temporal attention module from https://dl.acm.org/doi/abs/10.1145/3448083
    """

    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))

        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.dropout(x)
        outputs, h = self.rnn(x)
        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


class AttendAndDiscriminate(BaseModel):
    def __init__(
        self,
        input_dim,
        num_class,
        dataset,
        model="AttendAndDiscriminate",
        hidden_dim=128,
        filter_num=64,
        filter_size=5,
        enc_num_layers=2,
        enc_is_bidirectional=False,
        dropout=0.5,
        dropout_rnn=0.5,
        dropout_cls=0.5,
        activation="ReLU",
        sa_div=1,
        experiment="Default",
        isdeeper=False,
        timestamp=None,
        as_backbone=False,
    ):
        if timestamp is None:
            log_date = time.strftime("%Y%m%d")
            log_timestamp = time.strftime("%H%M%S")
            timestamp = f"{log_date}/{log_timestamp}"

        super(AttendAndDiscriminate, self).__init__(
            dataset=dataset, model=model, experiment=experiment, timestamp=timestamp
        )

        self.experiment = experiment
        self.model = model
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        print(paint(f"Creating {self.model} HAR model ..."))

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, num_class)
        self.register_buffer("centers", (torch.randn(num_class, self.hidden_dim)))

        self.as_backbone = as_backbone
        self.out_channels = hidden_dim

    def forward(self, x):
        feature = self.fe(x)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.classifier(out)

        if self.as_backbone:
            # For pre-training
            return z

        return z, logits

    def load_pretrained_weights(self, saved_model_path):
        # state_dict_path = os.path.join(saved_model_path)
        if not os.path.exists(saved_model_path):
            raise ValueError("Invalid path to the saved model")
        print("Loading the pre-trained weights at", saved_model_path)
        checkpoint = torch.load(saved_model_path)
        pretrained_checkpoint = checkpoint["model_state_dict"]

        # self.load_state_dict(pretrained_checkpoint, False)
        self.load_state_dict(pretrained_checkpoint, strict=True)
        return
