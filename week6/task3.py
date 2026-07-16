"""Week 6 Task 3: practice basic PyTorch tensor operations."""

import torch


def main() -> None:
    # Task 3-1: build a tensor from a Python list
    print("------ Task 3-1 ------")
    tensor = torch.tensor([[2, 3, 1], [5, -2, 1]])
    print(tensor.dtype, tensor.shape)

    # Task 3-2: build a 3x4x2 tensor filled with random floats from 0 to 1
    print("------ Task 3-2 ------")
    tensor = torch.rand((3, 4, 2))
    print(tensor.shape)
    print(tensor)

    # Task 3-3: build a 2x1x5 tensor filled with 1
    print("------ Task 3-3 ------")
    tensor = torch.ones((2, 1, 5))
    print(tensor.shape)
    print(tensor)

    # Task 3-4: calculate the matrix multiplication
    print("------ Task 3-4 ------")
    tensor_a = torch.tensor([[1, 2, 4], [2, 1, 3]])
    tensor_b = torch.tensor([[5], [2], [1]])
    print(torch.matmul(tensor_a, tensor_b))

    # Task 3-5: calculate the element-wise product
    print("------ Task 3-5 ------")
    tensor_a = torch.tensor([[1, 2], [2, 3], [-1, 3]])
    tensor_b = torch.tensor([[5, 4], [2, 1], [1, -5]])
    print(torch.mul(tensor_a, tensor_b))


if __name__ == "__main__":
    main()
