import torch

"""https://www.learnpytorch.io/00_pytorch_fundamentals/"""

# Scalar
scalar = torch.tensor(7)
print(scalar, " ", scalar.ndim)
# Get the Python number within a tensor (only works with one-element tensors)
print(scalar.item())


# Vector
vector = torch.tensor([7, 7])
print(vector)
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
print(TENSOR.ndim, "one three by three matrix")
print(TENSOR.shape)

random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

random_image_size_tensor = torch.rand(size=(224, 224, 3)) # 3 channels (RGB) of 224x224 pixels
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)    

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)

# Use torch.arange(), torch.range() is deprecated (veraltet)  
# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)

# Creating tensors like
ten_zeroes = torch.zeros_like(input=zero_to_ten)
print(ten_zeroes)

tensor = torch.tensor([1, 2, 3])
print(tensor.shape)
# Element-wise matrix multiplication
print(tensor * tensor)
# Matrix multiplication
print(torch.matmul(tensor, tensor))

# Transose tensor
# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)
# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)
# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T)

# Finding the min, max, mean, sum, etc (aggregation)
# Create a tensor
x = torch.arange(0, 100, 10)
print(x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")

# Change tensor datatype
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)