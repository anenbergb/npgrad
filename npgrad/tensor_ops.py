from npgrad.engine import Tensor


def matmul(input: Tensor, other: Tensor):
    """
    Matrix product of two tensors.
    """
    input_exp = input.unsqueeze(-1)  # shape (N,m,n,1)
    other_exp = other.unsqueeze(-3)  # shape (N,1,n,p)
    prod = input_exp * other_exp  # shape (N,m,n,p)
    return prod.sum(dim=-2)  # shape (m,p)
