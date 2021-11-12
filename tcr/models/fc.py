"""
Fully connected models
"""

from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedLayer(nn.Module):
    """Single fully connected layer"""

    def __init__(
        self, n_input: int, n_output: int, activation: Optional[nn.Module] = None
    ):
        super().__init__()
        self.n_in = n_input
        self.n_out = n_output
        self.fc = nn.Linear(self.n_in, self.n_out, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        self.act = activation

    def forward(self, x) -> torch.Tensor:
        x = self.fc(x)
        if self.act is not None:
            x = self.act(x)
        return x


class FullyConnectedNet(nn.Module):
    """
    Series of fully connected layers
    Example usage: FullyConnectedNet([100, 50, 20, 5])
    """

    def __init__(
        self,
        shapes: Iterable[int],
        activation: Callable = nn.PReLU,
        final_activation: Callable = nn.Sigmoid,
        seed: int = 1234,
    ):
        torch.manual_seed(seed)
        super().__init__()
        self.fc_layers = nn.ModuleList()
        for i, shape_pair in enumerate(zip(shapes[:-1], shapes[1:])):
            if i != len(shapes) - 2:
                self.fc_layers.append(
                    FullyConnectedLayer(*shape_pair, activation=activation())
                )
            else:
                self.fc_layers.append(
                    FullyConnectedLayer(*shape_pair, activation=final_activation())
                )

    def forward(self, x) -> torch.Tensor:
        for layer in self.fc_layers:
            x = layer(x)
        return x


def main():
    """On the fly testing"""
    pass


if __name__ == "__main__":
    main()
