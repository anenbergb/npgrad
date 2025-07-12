from typing import Optional, Union, List
import numpy as np


class Tensor:
    """
    stores a numpy tensor and its gradient

    - only support floating point data type
    """

    def __init__(self, data, _children=(), _op=""):
        if isinstance(data, np.ndarray):
            if data.dtype not in (np.float16, np.float32, np.float64):
                raise ValueError(f"Unsupported data type: {data.dtype}. Only support float 16, 32, 64")
            self.data = data
        else:
            self.data = np.array(data, dtype=np.float64)

        self.grad = np.zeros_like(self.data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # broadcast may be required
        # broadcast will add 1s to the left if to match shape
        # addition will fail and raise an error if broadcast shapes are incompatible
        out = Tensor(self.data + other.data, (self, other), "+")

        # For each dimension where the shape is 1 or the dimension was missing,
        # we sum the gradient of the output along that dimension
        def _backward():
            self.grad += reduce_grad(out.grad, self.data.shape)
            other.grad += reduce_grad(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        # broadcast other.data * out.grad, then reduce to self.data.shape
        def _backward():
            self.grad += reduce_grad(other.data * out.grad, self.data.shape)
            other.grad += reduce_grad(self.data * out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            dim = tuple(range(self.data.ndim))
        out_data = self.data.sum(axis=dim, keepdims=keepdim)
        out = Tensor(out_data, (self,), "sum")

        def _backward():
            # add back the collapsed dimensions
            grad = out.grad if keepdim else np.expand_dims(out.grad, axis=dim)
            # broadcast to match self.data.shape
            broadcast_grad = np.broadcast_to(grad, self.data.shape)
            self.grad += broadcast_grad

        out._backward = _backward

        return out

    def relu(self):
        out_data = self.data.copy()
        out_data[out_data < 0] = 0
        out = Tensor(out_data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        if self.numel() > 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    @property
    def shape(self):
        return self.data.shape

    def numel(self):
        return self.data.size

    def item(self):
        return self.data.item()

    def ones_like(self):
        return Tensor(np.ones_like(self.data))

    def zeros_like(self):
        return Tensor(np.zeros_like(self.data))


def reduce_grad(grad_output, original_shape):
    """
    Sum grad_output back to the shape of original_shape
    (used when original input was broadcast).
    """
    # Pad shape with 1s on the left if needed
    padded_shape = (1,) * (grad_output.ndim - len(original_shape)) + original_shape
    reduce_dims = tuple(i for i, orig in enumerate(padded_shape) if orig == 1)
    return grad_output.sum(axis=reduce_dims, keepdims=True).reshape(original_shape)
