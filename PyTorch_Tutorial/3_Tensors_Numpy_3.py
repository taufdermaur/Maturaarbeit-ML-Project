import torch
import numpy as np

# NumPy array to PyTorch tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).type(torch.float32) # .type() is used to convert the data type of the tensor; Numpy arrays are usually float64 by default
print(array, tensor)

# Change the value of array to see if it changes the tensor
array = array + 1
print(array, tensor) # The tensor is not changed because it is a copy of the array, not a reference to it

# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print("\n")
print(tensor, numpy_tensor)

# Change the value of tensor to see if it changes the numpy array
tensor = tensor + 1
print(tensor, numpy_tensor) # The numpy array is not changed because it is a copy of the tensor, not a reference to it
