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


# class Neuron(Module):
#     def __init__(self, nin, nonlin=True):
#         self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
#         self.b = Tensor(0)
#         self.nonlin = nonlin

#     def __call__(self, x):
#         act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
#         return act.relu() if self.nonlin else act

#     def parameters(self):
#         return self.w + [self.b]

#     def __repr__(self):
#         return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


# class Layer(Module):
#     def __init__(self, nin, nout, **kwargs):
#         self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

#     def __call__(self, x):
#         out = [n(x) for n in self.neurons]
#         return out[0] if len(out) == 1 else out

#     def parameters(self):
#         return [p for n in self.neurons for p in n.parameters()]

#     def __repr__(self):
#         return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


# class MLP(Module):
#     def __init__(self, nin, nouts):
#         sz = [nin] + nouts
#         self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]

#     def __repr__(self):
#         return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
