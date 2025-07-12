import torch
import numpy as np
from npgrad.engine import Tensor
from npgrad.layers import Linear

def test_linear():
    N, d_in, d_out = 5,4,3
    
    linear = Linear(d_in, d_out, bias=True, dtype=np.float32)
    x_np = np.random.randn(N, d_in).astype(np.float32)
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
    x_np = np.random.randn(N1,N2,d_in).astype(np.float32)
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
