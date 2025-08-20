import torch.nn as nn
from simuclr.utils import makedir


class BaseModel(nn.Module):
    def __init__(self, dataset, model, timestamp, experiment="default"):
        super(BaseModel, self).__init__()

        self.model = model
        self.dataset = dataset
        self.timestamp = timestamp
        self.experiment = experiment

        self.path_checkpoints = f"./logs/{self.model}/{self.dataset}/{self.experiment}/{self.timestamp}/checkpoints/"
        self.path_logs = f"./logs/{self.model}/{self.dataset}/{self.experiment}/{self.timestamp}/logs/"
        self.path_visuals = f"./logs/{self.model}/{self.dataset}/{self.experiment}/{self.timestamp}/visuals/"

    @property
    def path_checkpoints(self):
        return self._path_checkpoints

    @path_checkpoints.setter
    def path_checkpoints(self, path):
        makedir(path)
        self._path_checkpoints = path

    @property
    def path_logs(self):
        return self._path_logs

    @path_logs.setter
    def path_logs(self, path):
        makedir(path)
        self._path_logs = path

    @property
    def path_visuals(self):
        return self._path_visuals

    @path_visuals.setter
    def path_visuals(self, path):
        makedir(path)
        self._path_visuals = path
