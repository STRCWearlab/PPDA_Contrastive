import torch.nn as nn
from models.base_model import BaseModel
import time
import os
import torch


class DeepConvLSTM(BaseModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        dataset,
        experiment="default",
        conv_kernels=64,
        kernel_size=5,
        lstm_units=128,
        lstm_layers=2,
        timestamp=None,
        model="DeepConvLSTM",
        as_backbone=False,
    ):
        if timestamp is None:
            log_date = time.strftime("%Y%m%d")
            log_timestamp = time.strftime("%H%M%S")
            timestamp = f"{log_date}/{log_timestamp}"

        super(DeepConvLSTM, self).__init__(
            dataset=dataset, model=model, experiment=experiment, timestamp=timestamp
        )

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            n_channels * conv_kernels, lstm_units, num_layers=lstm_layers
        )

        self.classifier = nn.Linear(lstm_units, n_classes)

        self.activation = nn.ReLU()
        self.as_backbone = as_backbone
        self.out_channels = lstm_units  # for compatibility with SimCLR class

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.as_backbone:
            return x

        out = self.classifier(x)
        return x, out

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
