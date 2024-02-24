"""
Code modified from: https://github.com/eldrin/MTLMusicRepresentation-PyTorch/blob/master/musmtl/train.py#L22
Original paper: https://arxiv.org/abs/1802.04051
"""


from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cached_property import cached_property


class VGGLikeMultiTaskEncoder(nn.Module):
    """VGG-like architecture for Multi-Task Learning"""

    def __init__(self, tasks: List[str], n_outs: int = 128, n_ch_in: int = 2) -> None:
        super().__init__()
        self.n_outs: int = n_outs
        self.n_ch_in: int = n_ch_in
        self.tasks: List[str] = tasks

    @cached_property
    def encoder(self) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock2d(self.n_ch_in, 16, 5, conv_stride=(2, 1), pool_size=2),
            ConvBlock2d(16, 32, 3),
            ConvBlock2d(32, 64, 3),
            ConvBlock2d(64, 64, 3),
            ConvBlock2d(64, 128, 3),
            ConvBlock2d(128, 256, 3, pool_size=None),
            ConvBlock2d(256, 256, 1, pool_size=None),
            GlobalAveragePool(),
            linear_with_glorot_uniform(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
        )

    @cached_property
    def head(self) -> Dict[str, nn.Sequential]:
        return {
            task: nn.Sequential(
                linear_with_glorot_uniform(256, self.n_outs),
            )
            for task in self.tasks
        }

    def forward(self, X: torch.Tensor, task: Optional[str] = None) -> torch.Tensor:
        X = self.encoder(X)
        if task is not None:
            X = self.head[task](X)
        return X

    def eval(self):
        return self.encoder.eval()


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        pool_size: Optional[int] = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            conv_kernel,
            stride=conv_stride,
            padding=conv_kernel // 2,
        )
        nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channel)
        self.pool_size = pool_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.relu(self.bn(self.conv(X)))
        if self.pool_size is not None:
            X = F.max_pool2d(X, self.pool_size)
        return X


class GlobalAveragePool(nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.mean(X.view(X.size(0), X.size(1), -1), dim=2)


class SpecStandardScaler(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-10) -> None:
        super().__init__()
        self.mu = nn.Parameter(
            torch.from_numpy(mean)[None, None, None, :].float(), requires_grad=False
        )
        self.sigma = nn.Parameter(
            torch.max(torch.from_numpy(std).float(), torch.tensor([eps]))[
                None, None, None, :
            ],
            requires_grad=False,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mu) / self.sigma


def linear_with_glorot_uniform(f_in: int, f_out: int) -> nn.Linear:
    lin = nn.Linear(f_in, f_out)
    nn.init.xavier_uniform_(lin.weight)
    return lin
