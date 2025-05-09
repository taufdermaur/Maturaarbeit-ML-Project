what_were_covering = {1: "data (prepare and load)",
                      2: "build model",
                      3: "train model",
                      4: "making predictions and evaluating model (inference)",
                      5: "save and load model",
                      6: "putting it all together"}

import torch
from torch import nn # nn contains all of PyTorch's building blocks for machine learning
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("\n-----------------------------------------------------------------------------------------------------")

# 1. Data (prepare and load)

# Create know parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) 
y = weight * X + bias # Linear regression formula

# Visualize the data
print(X[:10], "\n", y[:10])
print(len(X), len(y)) 
print("\n-----------------------------------------------------------------------------------------------------")

# Splitting data into training an test sets
train_split = int(0.8 * len(X)) # 80% of the data for training
X_train, y_train = X[:train_split], y[:train_split] 
X_test, y_test = X[train_split:], y[train_split:] 

print(len(X_train), len(y_train), len(X_test), len(y_test))
print("\n-----------------------------------------------------------------------------------------------------")

def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions=None): # Plots training data, test data and compares predictions.
  
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  # Are there predictions?
  if predictions is not None:
    # Plot the predictions if they exist
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
  
  # Show the legend
  plt.legend(prop={"size": 14});

  plt.show()
  return plt.gcf()

"""plot_predictions()"""

# 2. Build model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 
                                           requires_grad=True, 
                                           dtype=torch.float
                                           )) # start with a random weight
        self.bias = nn.Parameter(torch.randn(1,
                                         requires_grad=True, 
                                         dtype=torch.float
                                         )) # start with a random bias
    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" is the input data
        return self.weight * x + self.bias # Linear regression formula


"""PyTorch model building essentials:
https://github.com/mrdbourke/pytorch-deep-learning/blob/main/video_notebooks/01_pytorch_workflow_video.ipynb"""

# Checking the contents of our PyTorch model

torch.manual_seed(42) # Set the random seed for reproducibility
model_0 = LinearRegressionModel()

# Check out the parameters
print(list(model_0.parameters())) # List of parameters in the model

# List named parameters
print(model_0.state_dict()) 
print("\n-----------------------------------------------------------------------------------------------------")

# Making predictions using "torch.inference_mode()"
with torch.inference_mode(): # This will turn off gradient tracking, so we can make predictions faster
  y_preds = model_0(X_test)
  print(y_preds)

"""plot_predictions(predictions=y_preds)"""
print("\n-----------------------------------------------------------------------------------------------------")

# Setup a loss function 
loss_fn = nn.L1Loss() # (mean absolute error)

# Setup a optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # Stochastic Gradient Descent
                            lr=0.01) # Learning rate (lr) of 0.01

# Building a training loop (and a testing loop)
