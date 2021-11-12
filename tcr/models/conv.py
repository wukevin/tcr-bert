"""
Convolutional models
"""

import os, sys
from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import fc


class ConvBatchNorm(nn.Module):
    """
    Convolutional layer, including nonlinearity, batchnorm
    Intended as a building block for larger networks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        # Convolutions expect (N, channels_in, length_in)
        # Convolutions ouptut (N, channels_out, length_out)
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = activation

    def forward(self, x) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvNet(nn.Module):
    """
    Convolutional network with a final global pooling
    This is useful for analyzing say variable sized sequence
    This is very loosely based on the Basset architecture, before the
    final convolutional layers

    Note that we severely reduce the number of layers and kernels due to the short
    nature of these sequences. With the default sizes, this returns shapes of:
    - [21, 16, 8] channels, [6, 6] channels -> (batch, 8, 26)
    """

    def __init__(
        self,
        channels: List[int] = [21, 16, 8],
        kernel_sizes: List[int] = [6, 6],
        pool_sizes: Optional[List[int]] = None,
    ):
        assert len(channels) - 1 == len(kernel_sizes)
        if pool_sizes is not None:
            assert len(kernel_sizes) == len(pool_sizes)

        super().__init__()
        self.layers = nn.ModuleList()
        for i, channel_pair in enumerate(zip(channels[:-1], channels[1:])):
            conv = ConvBatchNorm(*channel_pair, kernel_size=kernel_sizes[i])
            self.layers.append(conv)
            if pool_sizes:  # If no pooling, skip this
                pool = nn.MaxPool1d(pool_sizes[i])
                self.layers.append(pool)
        self.pool_sizes = pool_sizes
        self.kernel_sizes = kernel_sizes
        self.final_channels = channels[-1]

    def output_shape(self, input_length: int):
        if self.pool_sizes is not None:
            raise NotImplementedError
        ret_len = input_length
        for k in self.kernel_sizes:
            ret_len -= k - 1
        return self.final_channels, ret_len

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNetWithEmbedding(nn.Module):
    """
    Convolutional network, but wih an embedding layer instead of directly
    off of one-hot encoding
    Automatically treats the last index (default, 21st) as null embedding
    to use as padding
    With default chanels and kernel sizes, this returns shapes of:
    - [16, 16, 8] channels, [6, 6] kernel -> (batch, 8, 26)
    """

    def __init__(
        self,
        num_embed: int = 21,
        embed_dim: int = 16,
        channels: List[int] = [16, 32, 32, 16],
        kernel_sizes: List[int] = [5, 3, 3],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embed + 1, embed_dim, padding_idx=num_embed)
        self.layers = nn.ModuleList()
        for i, channel_pair in enumerate(zip(channels[:-1], channels[1:])):
            conv = ConvBatchNorm(*channel_pair, kernel_size=kernel_sizes[i])
            self.layers.append(conv)
        self.final_channels = channels[-1]
        self.kernel_sizes = kernel_sizes

    def output_shape(self, input_length: int):
        ret_len = input_length
        for k in self.kernel_sizes:
            ret_len -= k - 1
        return self.final_channels, ret_len

    def forward(self, x) -> torch.Tensor:
        # Embedding returns (batch, length, channels)
        # But convolution wants (batch, channels, length)
        x = self.embedding(x).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        return x


class OnePartConvNet(nn.Module):
    """
    One part Convnet, designed to look at either TRA or TRB only
    kwargs are passed to the convnet
    """

    def __init__(
        self,
        n_output: int,
        use_embedding: bool = False,
        max_input_len: int = 20,
        **kwargs
    ):
        super().__init__()
        if use_embedding:
            self.conv = ConvNetWithEmbedding(**kwargs)
        else:
            self.conv = ConvNet(**kwargs)

        self.n_output = n_output
        self.final_fc = fc.FullyConnectedLayer(
            np.prod(self.conv.output_shape(max_input_len)),
            self.n_output,
        )

    def forward(self, seq) -> torch.Tensor:
        bs = seq.shape[0]
        enc = self.conv(seq).reshape(bs, -1)
        return self.final_fc(enc)


class TwoPartConvNet(nn.Module):
    """
    Two part ConvNet, one part each for TCR-A/TCR-B sequence
    kwargs are passed to the convnet
    """

    def __init__(self, n_output: int = 2, use_embedding: bool = False, **kwargs):
        super().__init__()
        if use_embedding:
            # TODO share embedding layer?
            self.conv_a = ConvNetWithEmbedding(**kwargs)
            self.conv_b = ConvNetWithEmbedding(**kwargs)
        else:
            self.conv_a = ConvNet(**kwargs)
            self.conv_b = ConvNet(**kwargs)
        self.final_fc = fc.FullyConnectedLayer(
            np.prod(self.conv_a.output_shape(19))
            + np.prod(self.conv_b.output_shape(20)),
            n_output,
            # activation=nn.Softmax(dim=-1),
        )

    def forward(self, tcr_a, tcr_b) -> torch.Tensor:
        bs = tcr_a.shape[0]
        a_enc = self.conv_a(tcr_a).reshape(bs, -1)  # (bs, 208)
        b_enc = self.conv_b(tcr_b).reshape(bs, -1)
        enc = torch.cat([a_enc, b_enc], dim=-1)
        return self.final_fc(enc)


def main():
    """On the fly testing"""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import data_loader as dl

    net = TwoPartConvNet(use_embedding=True)
    print(net)

    table = dl.load_lcmv_table()
    dset = dl.TcrABSupervisedIdxDataset(table)
    x = net(dset[0][0]["tcr_a"].reshape(1, -1), dset[0][0]["tcr_b"].reshape(1, -1))
    print(x)


if __name__ == "__main__":
    main()
