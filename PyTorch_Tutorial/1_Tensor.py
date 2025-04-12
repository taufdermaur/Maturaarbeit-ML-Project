import torch

"""https://www.learnpytorch.io/00_pytorch_fundamentals/"""

# Scalar
scalar = torch.tensor(7)
print(scalar, " ", scalar.ndim)
# Get the Python number within a tensor (only works with one-element tensors)
print(scalar.item())


# Vector
vector = torch.tensor([7, 7])
vector
# Check the number of dimensions of vector
print(vector.ndim)
# Check shape of vector
print(vector.shape)


# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
print(MATRIX)
# Check number of dimensions
print(MATRIX.ndim) 
print(MATRIX.shape)#We get the output torch.Size([2, 2]) because MATRIX is two elements deep and two elements wide.


# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)    

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)