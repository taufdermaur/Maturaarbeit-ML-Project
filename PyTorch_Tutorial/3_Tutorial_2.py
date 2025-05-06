import torch
import pandas
import matplotlib.pyplot as plt
import numpy as np

print("Pytorch Version ", torch.__version__)

# create a line break
print("\n")  

# Create a tensor
some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Datatype: {some_tensor.dtype}") # the f stands for formatted string literals (f-strings) and allows you to embed expressions inside string literals, using curly braces { }.
print(f"Device: {some_tensor.device}")   
print(f"Shape; {some_tensor.shape}") 