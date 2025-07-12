import math
import numpy as np
from npgrad.engine import Tensor
from npgrad.numpy_ops import trunc_normal
from npgrad.tensor_ops import matmul


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: np.dtype = np.float32):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.empty((out_features, in_features), dtype=dtype))
        self.bias = Tensor(np.zeros((out_features,), dtype=dtype)) if bias else None
        # truncated Xavier init
        # balances the forward and backward pass variances
        variance = 2 / (in_features + out_features)
        std = math.sqrt(variance)
        trunc_normal(self.weight.shape, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass of the linear layer
        :param x: Input tensor of shape (batch_size, in_features)
        :return: Output tensor of shape (batch_size, out_features)
        """
        # (N, d_in) @ (d_out, d_in).T -> (N, d_out)

        out = matmul(x, self.weight.transpose(1, 0))
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None})"


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation="relu",
        bias: bool = True,
        dtype: np.dtype = np.float32,
    ):
        """
        input_dim: int, size of input features
        hidden_dims: list of int, hidden layer sizes
        output_dim: int, size of output features
        activation: relu, tanh
        """
        assert activation in ("relu", "tanh")
        self.activation = activation

        # Build a sequence of layers
        self.layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(Linear(prev_dim, h_dim, bias, dtype))
            prev_dim = h_dim

        # Output layer
        self.layers.append(Linear(prev_dim, output_dim, bias, dtype))

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == "relu":
                x = x.relu()
            elif self.activation == "tanh":
                x = x.tanh()
        return self.layers[-1](x)
