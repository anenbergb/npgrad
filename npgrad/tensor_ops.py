from npgrad.engine import Tensor


def matmul(input: Tensor, other: Tensor):
    """
    Matrix product of two tensors.
    """
    input_exp = input.unsqueeze(-1)  # shape (m,n,1)
    other_exp = other.unsqueeze(0)  # shape (1,n,p)
    prod = input_exp * other_exp  # shape (m,n,p)
    return prod.sum(dim=1)  # shape (m,p)
