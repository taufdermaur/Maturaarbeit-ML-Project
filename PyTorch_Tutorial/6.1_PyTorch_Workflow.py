what_were_covering = {1: "data (prepare and load)",
                      2: "build model",
                      3: "train model",
                      4: "making predictions and evaluating model (inference)",
                      5: "save and load model",
                      6: "putting it all together"}

import torch
from torch import nn # nn contains all of PyTorch's building blocks for machine learning
import matplotlib.pyplot as plt
import numpy as np

print("PyTorch version:", torch.__version__)
print("\n-----------------------------------------------------------------------------------------------------")

"""1. Data (prepare and load)"""

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

"""2. Build model"""

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

"""3. Train model"""
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu" 
torch.manual_seed(42) 
model_0 = LinearRegressionModel().to(device)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Setup a loss function 
loss_fn = nn.L1Loss() # (mean absolute error)

# Setup a optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # Stochastic Gradient Descent
                            lr=0.01) # Learning rate (lr) of 0.01

# Building a training loop (and a testing loop)
epochs = 168 # an epoch is one loop through the data (this is a hyperparameter)

# Track different values for comparison
epoch_count = []
loss_values = []  
test_loss_values = []

print("Starting training loop:")
print(f"Epoch 0 | Initial parameters: {model_0.state_dict()}")
print("\n-----------------------------------------------------------------------------------------------------")

# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train() # sets all parameters that require to require gradients

    # 1. Forward pass
    y_pred = model_0(X_train) # Pass the training data through the model

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train) # Compare the predictions to the true labels
    print(f"Loss: {loss}")

    # 3. Optimizer zero grad
    optimizer.zero_grad() 

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward() 

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step() 
    # by default how the optimizer changes will accumulate thorugh the loop so we have to zero them above in step 3 for the next iteration of the loop

    print(f"Epoch {epoch+1} | Loss: {loss:.6f} | Weight: {model_0.weight.item():.6f}, Bias: {model_0.bias.item():.6f}")

    """4. making predictions and evaluating the model (inference)"""
    model_0.eval() # turns off different settings in the model not needed for evaluation/testing (dropout, batch norm, etc.)

    with torch.inference_mode(): # turns off gradient tracking
      # 1. Forward pass
      test_pred = model_0(X_test)

      # 2. Calculate the loss
      test_loss = loss_fn(test_pred, y_test)

    print(f"Test Loss: {test_loss:.6f}")
    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)
    print("-----------------------------------------------------------------------------------------------------")

print("\nTraining complete!")
print(f"Final parameters: {model_0.state_dict()}")
print(f"True parameters: Weight = {weight}, Bias = {bias}")
print("\n-----------------------------------------------------------------------------------------------------")

with torch.inference_mode():
    y_preds_new = model_0(X_test)
    # Move tensors to CPU for plotting
    plot_predictions(
        train_data=X_train.cpu(),
        train_labels=y_train.cpu(),
        test_data=X_test.cpu(),
        test_labels=y_test.cpu(),
        predictions=y_preds.cpu()
    )

plt.plot(epoch_count, np.array(torch.tensor(loss_values).cpu().numpy()), label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.title("training and test loss curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plot_predictions(predictions=y_preds_new.cpu())

"""5. Save and load model"""

from pathlib import Path

# 1. Create a model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

# 2. Create a model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME # Path to save the model
print(f"Saving model to: {MODEL_SAVE_PATH}")

# 3. Save the model state dict
torch.save(obj=model_0.state_dict(), # Save the model parameters
           f=MODEL_SAVE_PATH) # Save the model to the path

# 4. Load the model
loaded_model_0 = LinearRegressionModel() # Create a new instance of the model

# 5. Load the model state dict (this will overwrite the parameters of the new model)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) 
print(loaded_model_0.state_dict())
print("\n-----------------------------------------------------------------------------------------------------")

# Making some predictions with the loaded model
loaded_model_0.eval() # Set the model to evaluation mode
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

model_0.eval() # Set the model to evaluation mode
with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds == loaded_model_preds) # the predictions are the same  
