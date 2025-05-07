# Trying to take random out of radom
import torch

print(torch.rand(3, 3)) # this will always give you a different result
print("\n-----------------------------------------------------------------------------------------------------")

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

print("\n-----------------------------------------------------------------------------------------------------")
# Create random but reproducible tensors
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED) # torch.manual_seed() only works for one block of code
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

print("\n-----------------------------------------------------------------------------------------------------")