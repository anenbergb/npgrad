import numpy as np
import torch


from npgrad.engine import Tensor
from npgrad.tensor_ops import matmul

def test_matmul():
    np1 = np.arange(4, dtype=np.float64).reshape((2,2))
    np2 = np.arange(6, dtype=np.float64).reshape((2,3))

    tensor1 = Tensor(np1)
    tensor2 = Tensor(np2)
    tensor3 = matmul(tensor1, tensor2)
    sum_np  = tensor3.sum()
    sum_np.backward()

    tensor1pt = torch.from_numpy(np1)
    tensor1pt.requires_grad = True
    tensor2pt = torch.from_numpy(np2)
    tensor2pt.requires_grad = True
    tensor3pt = torch.matmul(tensor1pt, tensor2pt)
    sumpt  = tensor3pt.sum()
    sumpt.backward()

    assert np.allclose(sum_np.data, sumpt.detach().numpy())
    assert np.allclose(tensor1.grad, tensor1pt.grad.numpy())
    assert np.allclose(tensor2.grad, tensor2pt.grad.numpy())
