# Indexing (selecting data from tensors)

# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1, 3, 3) 
print(x, x.shape)

print("\nx[0] = ", x[0]) # dim=0
print("\nx[0][0] = ", x[0][0]) # dim=1
print("\nx[0][0][0] = ", x[0][0][0]) # dim=2

# print("\nx[1][1][1]() = ", x[0][0][0]) will raise an error because there is no index 1 in dim=0

# Challange: find the value 9 in the tensor
print("\nx[0][2][2] = ", x[0][2][2]) 

# You can also use ":" to select "all" of a target dimension
print("\nx[:, 0] = ", x[:, 0]) 

print("\nx[:, :, 1] = ", x[:, :, 1]) # all values of the dim=0 and dim=1 but only index 1 of dim=2

print("\nx[:, 1, 1] = ", x[:, 1, 1]) # all values of the dim=0 but only index 1 of dim=1 and dim=2

print("\nx[0, 0, :] = ", x[0, 0, :]) # all values of the dim=2 but only index 0 of dim=0 and dim=1

# Challange: Index on x to return 3, 6, 9
print("\nx[:, :, 2] = ", x[:, :, 2]) # all values of the dim=0 and dim=1 but only index 2 of dim=2")
