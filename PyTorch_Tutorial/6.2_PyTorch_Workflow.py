"""6. Putting it all together"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("-----------------------------------------------------------------------------------------------------")

# Create some data using the linier regression formula
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step= 0.02

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) 
y = weight * X + bias 
print(X[:10], "\n", y[:10])

# Split the data into training and test sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test))
print("-----------------------------------------------------------------------------------------------------")

# Visualize the data
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

plot_predictions()

# Create the model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters / also called: Linear transfrom, probing layer etc. 
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1) # 1 input, 1 output 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set the manual seed for reproducibility
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1.to(device) 
print(model_1, "\n", "Starting model state dict: ", model_1.state_dict()) 
print("-----------------------------------------------------------------------------------------------------")

# Loss function
loss_fn = nn.L1Loss() 

# Optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                              lr=0.01) # Stochastic Gradient Descent with learning rate of 0.01

# Training loop
torch.manual_seed(42)

epochs = 200

# Put data on the target device
X_train = X_train.to(device)
y_train = y_train.to(device)   
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Testing 
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    # Print out whats happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}") 
        print("-----------------------------------------------------------------------------------------------------")

print("Final model state dict: ", model_1.state_dict())
print("-----------------------------------------------------------------------------------------------------")

# Make predictions
with torch.inference_mode():
    y_preds = model_1(X_test)
    print(f"Predictions: {y_preds}")
    print("-----------------------------------------------------------------------------------------------------")

# Plot predictions
plot_predictions(predictions=y_preds.cpu()) # Matplotlib works with numpy, so we need to move the tensor back to CPU

# Save the model
# 1. Create a model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True) 

# 2. Create a model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), 
           f=MODEL_SAVE_PATH) 

# Load the model
loaded_model_1 = LinearRegressionModelV2() 
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_1.to(device) 
next(loaded_model_1.parameters()).device # Check if the model is on the right device
print(f"Loaded model state dict: {loaded_model_1.state_dict()}")
print("-----------------------------------------------------------------------------------------------------")

# Evaluate the loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
print(y_preds == loaded_model_1_preds) 
