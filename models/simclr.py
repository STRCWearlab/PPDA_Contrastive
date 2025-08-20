import os
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.GRU, nn.LSTM)):
                print(m)
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        x = self.dropout(self.relu(self.conv(x)))
        return x


class SimCLR(nn.Module):
    """
    The base SimCLR model.
    """

    def __init__(self, backbone, device):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.device = device
        # self.backbone must have .out_channels attribute
        assert hasattr(
            self.backbone, "out_channels"
        ), "Backbone must have an out_channels attribute"

        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 50),
        )

        self.to(self.device)

    def forward(self, inputs):
        backbone = self.backbone(inputs)
        projection = self.projection_head(backbone)

        return projection

    def load_pretrained_weights(self, saved_model_path):
        if not os.path.exists(saved_model_path):
            raise ValueError("Invalid path to the saved model")
        print("Loading the pre-trained weights")
        checkpoint = torch.load(saved_model_path, map_location=self.device)
        pretrained_checkpoint = checkpoint["model_state_dict"]

        # self.load_state_dict(pretrained_checkpoint, False)
        # Only loading the backbone weights
        self.backbone.load_state_dict(pretrained_checkpoint, strict=True)
        return


class TPN(BaseModel):
    """
    Temporal Predictive Network (TPN). https://arxiv.org/pdf/1907.11879
    """

    def __init__(
        self,
        in_channels,
        out_channels=96,
        dataset=None,
        experiment="pre-train",
        timestamp=None,
        kernel_sizes=[24, 16, 8],
    ):
        if timestamp is None:
            log_date = time.strftime("%Y%m%d")
            log_timestamp = time.strftime("%H%M%S")
            timestamp = f"{log_date}/{log_timestamp}"

        super(TPN, self).__init__(
            dataset, "TPN", experiment=experiment, timestamp=timestamp
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvBlock(
            in_channels=self.in_channels, out_channels=32, kernel_size=kernel_sizes[0]
        )
        self.conv2 = ConvBlock(
            in_channels=32, out_channels=64, kernel_size=kernel_sizes[1]
        )
        self.conv3 = ConvBlock(
            in_channels=64, out_channels=self.out_channels, kernel_size=kernel_sizes[2]
        )

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global Max Pooling (as per
        # https://github.com/keras-team/keras/blob
        # /7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/layers/pooling.py
        # #L559) for 'channels_first'
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.squeeze(2)

        return x


class Classifier(BaseModel):
    def __init__(
        self,
        backbone,
        num_classes,
        device,
        dataset,
        classification_model="mlp",
        experiment="default",
        timestamp=None,
    ):
        if timestamp is None:
            log_date = time.strftime("%Y%m%d")
            log_timestamp = time.strftime("%H%M%S")
            timestamp = f"{log_date}/{log_timestamp}"
        super(Classifier, self).__init__(
            dataset, "Classifier", experiment=experiment, timestamp=timestamp
        )

        # Encoder
        self.backbone = backbone
        self.device = device
        if not isinstance(self.backbone, TPN):
            # Note: currently, the backbone is expected to be of TPN type
            # Otherwise, freeze_two_conv_layers() may not work as expected
            warnings.warn(
                "The backbone is not of TPN type. freeze_two_conv_layers() may not work as expected."
            )

        self.num_classes = num_classes
        self.classification_model = classification_model

        # Softmax
        if self.classification_model == "linear":
            self.softmax = nn.Linear(self.backbone.out_channels, num_classes)
        elif self.classification_model == "mlp":
            self.softmax = nn.Sequential(
                nn.Linear(self.backbone.out_channels, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            )
        else:
            raise ValueError("Invalid classification model")

    def forward(self, inputs):
        # Passing it through the encoder
        backbone = self.backbone(inputs)
        softmax = self.softmax(backbone)
        return None, softmax  # z, logits

    def load_pretrained_weights(self, saved_model_path):
        # state_dict_path = os.path.join(saved_model_path)
        if not os.path.exists(saved_model_path):
            raise ValueError("Invalid path to the saved model")
        print("Loading the pre-trained weights at", saved_model_path)
        checkpoint = torch.load(saved_model_path, map_location=self.device)
        pretrained_checkpoint = checkpoint["model_state_dict"]

        # self.load_state_dict(pretrained_checkpoint, False)
        # Only loading the backbone weights
        self.backbone.load_state_dict(pretrained_checkpoint, strict=True)
        return

    def freeze_encoder_layers(self):
        """
        To set only the softmax to be trainable
        :return: None, just setting the encoder part as frozen
        """
        # First setting the model to eval
        self.backbone.eval()

        # Then setting the requires_grad to False
        for param in self.backbone.parameters():
            param.requires_grad = False

        return

    def freeze_two_conv_layers(self):
        """
        Setting the first two conv layers to be frozen.
        Classifier and the last conv layer remain trainable.
        """
        # First setting the two conv layers to eval
        self.backbone.conv1.eval()
        self.backbone.conv2.eval()

        # Then setting the requires_grad to False
        for param in self.backbone.named_parameters():
            if "conv3" in param[0]:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False

        return

    def freeze_conv_layers_keep_lstm(self):
        """
        Freezes all convolutional layers while leaving the LSTM layers trainable.
        For DeepConvLSTM model.
        """
        # Set convolutional layers to eval mode and freeze parameters
        self.backbone.conv1.eval()
        self.backbone.conv2.eval()
        self.backbone.conv3.eval()
        self.backbone.conv4.eval()

        for name, param in self.backbone.named_parameters():
            if "lstm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return

    def freeze_attend_layers(self, keep_trainable=["ta", "rnn"]):
        """
        Freezes layers except those specified in keep_trainable.

        Args:
            keep_trainable (list): List containing names of layers to keep trainable.
                                   Options: 'ta' (Temporal Attention), 'rnn' (RNN layers),
                                   'sa' (Self-Attention), 'conv' (Convolutional layers).
        """
        layer_map = {
            "conv": ["fe.conv1", "fe.conv2", "fe.conv3", "fe.conv4"],
            "sa": ["fe.sa"],
            "rnn": ["fe.rnn"],
            "ta": ["fe.ta"],
        }

        # Set specified layers to eval mode and freeze parameters by default
        for group, layers in layer_map.items():
            for layer_name in layers:
                layer = eval(f"self.backbone.{layer_name}")
                if group not in keep_trainable:
                    layer.eval()

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            for key in keep_trainable:
                if key in name:
                    print(f"Keeping {name} trainable")
                    param.requires_grad = True

        return
