import torch

print("GPU available? ", torch.cuda.is_available()) # Check if GPU is available

# Set the device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device) 

print("\n")
# Putting tensor (and models) on the GPU
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device) 

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu) 

# If tensor is on GPU, can't transform it to NumPy
# tensor_on_gpu.numpy() # This will raise an error
tensor_back_on_cpu = tensor_on_gpu.to("cpu") # Move it back to CPU
print(tensor_back_on_cpu.numpy()) # Now we can convert it to NumPy