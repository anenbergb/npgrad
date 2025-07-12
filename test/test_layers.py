import torch
import numpy as np
from npgrad.engine import Tensor
from npgrad.layers import Linear, MLP

def test_linear():
    N, d_in, d_out = 5,4,3
    
    linear = Linear(d_in, d_out, bias=True, dtype=np.float32)
    x_np = np.random.uniform(low=-1.0, high=1.0, size = (N,d_in)).astype(np.float32)
    x = Tensor(x_np)
    out = linear(x)
    sum_np  = out.sum()
    sum_np.backward()

    linear_pt = torch.nn.Linear(d_in, d_out, bias=True, dtype=torch.float32)
    linear_pt.weight.data = torch.from_numpy(linear.weight.data)
    linear_pt.bias.data = torch.from_numpy(linear.bias.data)
    x_pt = torch.from_numpy(x_np)
    x_pt.requires_grad = True
    out_pt = linear_pt(x_pt)
    sumpt  = out_pt.sum()
    sumpt.backward()

    assert np.allclose(out.data, out_pt.detach().numpy())
    assert np.allclose(sum_np.data, sumpt.detach().numpy())
    assert np.allclose(linear.weight.grad, linear_pt.weight.grad.numpy())
    assert np.allclose(linear.bias.grad, linear_pt.bias.grad.numpy())
    assert np.allclose(x.grad, x_pt.grad.numpy())

def test_linear2():
    N1, N2, d_in, d_out = 6,5,4,3
    
    linear = Linear(d_in, d_out, bias=True, dtype=np.float32)
    x_np = np.random.uniform(low=-1.0, high=1.0, size = (N1,N2,d_in)).astype(np.float32)
    x = Tensor(x_np)
    out = linear(x)
    sum_np  = out.sum()
    sum_np.backward()

    linear_pt = torch.nn.Linear(d_in, d_out, bias=True, dtype=torch.float32)
    linear_pt.weight.data = torch.from_numpy(linear.weight.data)
    linear_pt.bias.data = torch.from_numpy(linear.bias.data)
    x_pt = torch.from_numpy(x_np)
    x_pt.requires_grad = True
    out_pt = linear_pt(x_pt)
    sumpt  = out_pt.sum()
    sumpt.backward()

    assert np.allclose(out.data, out_pt.detach().numpy())
    assert np.allclose(sum_np.data, sumpt.detach().numpy())
    assert np.allclose(linear.weight.grad, linear_pt.weight.grad.numpy())
    assert np.allclose(linear.bias.grad, linear_pt.bias.grad.numpy())
    assert np.allclose(x.grad, x_pt.grad.numpy())


class MLP_pt(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=torch.nn.ReLU, bias: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__()

        # Build a sequence of layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim, bias=bias, dtype=dtype))
            layers.append(activation())
            prev_dim = h_dim

        # Output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim, bias=bias, dtype=dtype))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def test_mlp():
    N = 5
    d_in = 10
    d_out = 3
    hidden_dims = [10,15,20]
    mlp = MLP(d_in, hidden_dims, d_out, activation="relu", bias=True, dtype=np.float32)
    x_np = np.random.uniform(low=-1.0, high=1.0, size = (N, d_in)).astype(np.float32)
    x = Tensor(x_np)
    out = mlp(x)
    sum_np  = out.sum()
    sum_np.backward()

    mlp_pt = MLP_pt(d_in, hidden_dims, d_out, activation=torch.nn.ReLU, bias=True, dtype=torch.float32)
    layer_idx = 0
    for layer_pt in mlp_pt.net:
        if isinstance(layer_pt, torch.nn.Linear):
            layer = mlp.layers[layer_idx]
            layer_pt.weight.data = torch.from_numpy(layer.weight.data)
            layer_pt.bias.data = torch.from_numpy(layer.bias.data)
            layer_idx += 1

    x_pt = torch.from_numpy(x_np)
    x_pt.requires_grad = True
    out_pt = mlp_pt(x_pt)
    sumpt  = out_pt.sum()
    sumpt.backward()

    assert np.allclose(out.data, out_pt.detach().numpy())
    assert np.allclose(sum_np.data, sumpt.detach().numpy())
    layer_idx = 0
    for layer_pt in mlp_pt.net:
        if isinstance(layer_pt, torch.nn.Linear):
            layer = mlp.layers[layer_idx]
            assert np.allclose(layer.weight.grad, layer_pt.weight.grad.numpy()), f"MLP Layer[{layer_idx}] weight.grad mismatch"
            assert np.allclose(layer.bias.grad, layer_pt.bias.grad.numpy()), f"MLP Layer[{layer_idx}] bias.grad mismatch"
            layer_idx += 1

    assert np.allclose(x.grad, x_pt.grad.numpy())