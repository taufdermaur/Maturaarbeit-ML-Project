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

# Tensor Datatypes
float_32_tensor = torch.tensor([1.0, 2.0, 3.0], 
                               dtype=None, # what datatype is the tensor (e.g. float32, int64, etc.)
                               device="cpu", # what device is your tensor on (e.g. cpu, cuda, None)
                               requires_grad=False) # whether or not to track gradients with this tensor
print(float_32_tensor, float_32_tensor.dtype)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor, float_16_tensor.dtype)

int_64_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
print(int_64_tensor, int_64_tensor.dtype)


print("\n") 
print("\n") 
# Tensor Operations (Manipulating Tensors)
tensor = torch.tensor([1, 2, 3])
print(tensor.shape)
print(tensor + 10) # Add 10 to each element in the tensor
print(tensor * 10) # Multiply each element in the tensor by 10

print(torch.mul(tensor, 10)) # Multiply each element in the tensor by 10

# Element-wise matrix multiplication
print(tensor * tensor)
# Matrix multiplication
print(torch.matmul(tensor, tensor))

# TRANSPOSE tensor
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

print(torch.matmul(tensor_A, tensor_B.T)) #If without ".T" then: RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)

print("\n")
print("\n")
print("\n")

# Finding the min, max, mean, sum, etc (aggregation)
# Create a tensor
x = torch.arange(1, 100, 10) # 1 to 100 with step 10
print(x, x.shape, x.dtype)

print(f"Minimum: {x.min()} or {torch.min(x)}")
print(f"Maximum: {x.max()} or {torch.max(x)}")

# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()} or {torch.mean(x.type(torch.float32))}") # won't work without float datatype; int64 won't work
print(f"Sum: {x.sum()} or {torch.sum(x)}")

# position of max and min value
print(f"Position of max value: {x.argmax()} or {torch.argmax(x)}") # returns the index of the max value
print(f"Position of min value: {x.argmin()} or {torch.argmin(x)}") # returns the index of the min value

print("\n")
print("\n")
print("\n")
# Reshaping, stacking, squeezing and unsqueezing tensors
y = torch.arange(1., 10.)
print(y, y.shape, y.ndim)

y_reshaped = y.reshape(1, 9) # reshape to 1 row and 9 columns
y_reshaped2 = y.reshape(9, 1) # reshape to 9 rows and 1 column

z = y.view(1, 9) # view the tensor as 1 row and 9 columns
print(z, z.shape, z.ndim) # view doesn't create a new tensor, it just changes the view of the original tensor

z[:, 0] = 5 # change the first element of z to 5
print(z, y) # this will also change the first element of y to 5 because z is just a view of y

y_stacked = torch.stack([y,y,y,y], dim=0) # stack the tensor along the first dimension (rows)
print(y_stacked)

print("\n")
print(y_reshaped)
y_reshaped_squeezed = y_reshaped.squeeze() # remove all single dimensions (1D) from a tensor
print(y_reshaped_squeezed)
# torch.Size([1, 9]) -> torch.Size([9]) here we remove the first dimension (1 row)

y_reshaped_unsqueezed = y_reshaped_squeezed.unsqueeze(dim=0)
print(y_reshaped_unsqueezed) # add a new dimension (1D) to the tensor

print("\n")
print("\n")
print("\n")
# Change tensor datatype
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)