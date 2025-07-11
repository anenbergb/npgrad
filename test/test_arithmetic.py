import torch
import numpy as np
from npgrad.engine import Tensor

def test_scalar():
    x = Tensor(4.0)
    x.backward()
    xpt = torch.tensor(4.0, requires_grad=True, dtype=torch.float64)
    xpt.backward()
    assert x.grad.item() == xpt.grad.item()

    x = Tensor([4.0])
    x.backward()
    xpt = torch.tensor([4.0], requires_grad=True, dtype=torch.float64)
    xpt.backward()
    assert x.grad == xpt.grad.numpy()

def test_sum():
    tensor1 = Tensor([[1,2],[3,4]])
    tensor2 = tensor1.ones_like()
    tensor3 = tensor1 + tensor2
    sum_np  = tensor3.sum()
    sum_np.backward()

    tensor1pt = torch.tensor([[1,2],[3,4]], requires_grad=True, dtype=torch.float64)
    tensor2pt = torch.ones_like(tensor1pt, requires_grad=True)
    tensor3pt = tensor1pt + tensor2pt
    sumpt  = tensor3pt.sum()
    sumpt.backward()

    assert sum_np.data == sumpt.detach().numpy()
    assert np.allclose(tensor1.grad, tensor1pt.grad.numpy())
    assert np.allclose(tensor2.grad, tensor2pt.grad.numpy())

def test_sum_broadcast():
    tensor1 = Tensor([[1,2],[3,4]])
    tensor2 = Tensor([2,1])
    tensor3 = tensor1 + tensor2
    sum_np  = tensor3.sum()
    sum_np.backward()

    tensor1pt = torch.tensor([[1,2],[3,4]], requires_grad=True, dtype=torch.float64)
    tensor2pt = torch.tensor([2,1,], requires_grad=True, dtype=torch.float64)
    tensor3pt = tensor1pt + tensor2pt
    sumpt  = tensor3pt.sum()
    sumpt.backward()

    assert sum_np.data == sumpt.detach().numpy()
    assert np.allclose(tensor1.grad, tensor1pt.grad.numpy())
    assert np.allclose(tensor2.grad, tensor2pt.grad.numpy())


# def test_sanity_check():

#     x = Tensor([-4.0])
#     z = 2 * x + 2 + x
#     q = z.relu() + z * x
#     h = (z * z).relu()
#     y = h + q + q * x
#     y.backward()
#     xmg, ymg = x, y

#     x = torch.Tensor([-4.0])
#     x.requires_grad = True
#     z = 2 * x + 2 + x
#     q = z.relu() + z * x
#     h = (z * z).relu()
#     y = h + q + q * x
#     y.backward()
#     xpt, ypt = x, y

#     # forward pass went well
#     assert ymg.data == ypt.data.item()
#     # backward pass went well
#     assert xmg.grad == xpt.grad.item()

# def test_more_ops():

#     a = Tensor([-4.0])
#     b = Tensor([2.0])
#     c = a + b
#     d = a * b + b**3
#     c += c + 1
#     c += 1 + c + (-a)
#     d += d * 2 + (b + a).relu()
#     d += 3 * d + (b - a).relu()
#     e = c - d
#     f = e**2
#     g = f / 2.0
#     g += 10.0 / f
#     g.backward()
#     amg, bmg, gmg = a, b, g

#     a = torch.Tensor([-4.0]).double()
#     b = torch.Tensor([2.0]).double()
#     a.requires_grad = True
#     b.requires_grad = True
#     c = a + b
#     d = a * b + b**3
#     c = c + c + 1
#     c = c + 1 + c + (-a)
#     d = d + d * 2 + (b + a).relu()
#     d = d + 3 * d + (b - a).relu()
#     e = c - d
#     f = e**2
#     g = f / 2.0
#     g = g + 10.0 / f
#     g.backward()
#     apt, bpt, gpt = a, b, g

#     tol = 1e-6
#     # forward pass went well
#     assert abs(gmg.data - gpt.data.item()) < tol
#     # backward pass went well
#     assert abs(amg.grad - apt.grad.item()) < tol
#     assert abs(bmg.grad - bpt.grad.item()) < tol