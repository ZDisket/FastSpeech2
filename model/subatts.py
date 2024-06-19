from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# subatts.py: normalization, activation functions

class APTxS1(nn.Module):
    """
    APTx Stage 1:

    - Trainable beta and gamma (allows model to dynamically adjust upwards slope and scaling)
    - Squaring output, inspired by Squared ReLU.

    Both of these modifications have proven to increase accuracy in small tests (4-layer encoder on IMDB)
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, trainable=False):
        super(APTxS1, self).__init__()
        self.alpha = alpha
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.beta = beta
            self.gamma = gamma

    def forward(self, x):
        return ((self.alpha + torch.tanh(self.beta * x)) * self.gamma * x) ** 2


class APTx(nn.Module):
    """
    APTx: Alpha Plus Tanh Times, an activation function that behaves like Mish,
    but is 2x faster.

    https://arxiv.org/abs/2209.06119
    """

    def __init__(self, alpha=1, beta=1, gamma=0.5, trainable=False):
        """
        Initialize APTx initialization.
        :param alpha: Alpha
        :param beta: Beta
        :param gamma: Gamma
        :param trainable: Makes beta and gamma trainable, dynamically optimizing the upwards slope and scaling
        """
        super(APTx, self).__init__()
        self.alpha = alpha
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.beta = beta
            self.gamma = gamma

    def forward(self, x):
        return (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x


class DPReLU(nn.Module):
    """
    DPReLU: A dynamic ReLU variant:

    "There are four additional learnable parameters compared to the vanilla ReLU. alpha and beta are the slopes of the negative
    and positive parts in the function, respectively. Here, a negative or positive case is determined when comparing input
    x to the threshold. The threshold makes DPReLU shift on the x-axis in comparison to the original ReLU. The bias
    determines the alignment of the function with respect to the y-axis. These four parameters are all learnable and interact
    with each other during the training phase"

    https://link.springer.com/article/10.1007/s44196-023-00186-w

    By default, alpha and beta are 0.5 and 0.9, which yielded best results according to the paper. Threshold and bias = 0.

    Important: Please use He or their custom initialization!

    Converted from Tensorflow from https://github.com/KienMN/Activation-Experiments/tree/master
    """

    def __init__(self, alpha_init=0.5, beta_init=0.9, threshold_init=0.0, bias_init=0.0, shared_axes=None):
        super(DPReLU, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))

        self.shared_axes = shared_axes
        if self.shared_axes is not None and not isinstance(self.shared_axes, (list, tuple)):
            self.shared_axes = [self.shared_axes]

    def forward(self, inputs):
        neg = -self.alpha * torch.relu(-inputs + self.threshold)
        pos = self.beta * torch.relu(inputs - self.threshold)
        return pos + neg + self.bias

    def extra_repr(self):
        return f'alpha={self.alpha.item()}, beta={self.beta.item()}, threshold={self.threshold.item()}, bias={self.bias.item()}, shared_axes={self.shared_axes}'


class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=None,
            drop=0.0,
            bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)



class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class TransposeRMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(TransposeRMSNorm, self).__init__()
        self.ln = RMSNorm(num_features, eps=eps)

    def forward(self, x):
        # Transpose from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Apply RMSNorm
        x = self.ln(x)
        # Transpose back from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        return x


class TransposeLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(TransposeLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(num_features, eps, affine)

    def forward(self, x):
        # Transpose from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Apply LayerNorm
        x = self.ln(x)
        # Transpose back from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        return x



class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Pass through simple SwiGLU
        :param x:
        :return:
        """
        x = x.transpose(1, 2)
        x_proj = self.fc1(x)
        x_proj, x_gate = x_proj.chunk(2, dim=-1)
        x = x_proj * torch.sigmoid(x_gate)
        return x.transpose(1, 2)



class SwiGLUCNN(nn.Module):
    def __init__(self):
        super(SwiGLUCNN, self).__init__()

    def forward(self, x):
        """
        :param x: input tensor of shape (batch_size, dim, seq_length)
        :return: output tensor of shape (batch_size, dim // 2, seq_length)
        """
        # Split the input tensor into two equal parts along the last dimension
        x = x.transpose(1, 2)
        x_proj, x_gate = x.chunk(2, dim=-1)
        # Apply the SwiGLU activation function
        x = x_proj * torch.sigmoid(x_gate)
        x = x.transpose(1, 2)
        return x
