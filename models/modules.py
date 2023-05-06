import torch.nn as nn
from torch import Tensor


class ConditionalBatchNorm2d(nn.Module):
    def __init__(
            self,
            num_classes: int,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        affine = False
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)
        nn.init.ones_(self.weights.weight)
        nn.init.zeros_(self.biases.weight)

    def forward(self, x: Tensor, y: Tensor):
        x = self.bn(x)
        weight = self.weights(y)[:, :, None, None]
        bias = self.biases(y)[:, :, None, None]
        return x * weight + bias
