'''
Simple Forward and Backward Passes in PyTorch
'''

import torch

class LinearManual(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias   = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(5, 3)  # batch of 5, input dim 3
    layer = LinearManual(3, 2)  # output dim 2

    print("Input:\n", x)
    print("Weight:\n", layer.weight)
    print("Bias:\n", layer.bias)
    
    # Define the optimizer
    learning_rate = 0.01

    for i in range(3):
        
        # Forward pass
        out = layer(x)
        print(f"\nOutput after forward pass {i+1}:\n", out)
        
        # Compute the loss (simple sum of outputs)
        loss = out.sum()
        print(f"Loss after forward pass {i+1}: {loss.item():.4f}")
        

        # Gradients (math)
        # Z = x @ W.T + b
        # L = Z.sum()
        # dL/dZ = 1 (since L is sum of all elements in Z
        # dL/dW = dL/dZ * dZ/dW = dZ.T @ x
        # dL/db = dL/dZ * dZ/db = dZ.sum(dim=0)
        # dZ = torch.ones(5, 2)        # dL/dZ = 1 everywhere (derivative of sum)
        # dW = dZ.T @ x                # (2, 3)
        # db = dZ.sum(dim=0)           # (2,)
        
        # Backward pass
        layer.zero_grad()  # Clear previous gradients
        loss.backward() # Compute gradients with respect to weights and bias
    
        # Print gradients
        print(f"  dW: {layer.weight.grad}")
        print(f"  db: {layer.bias.grad}")

        # Update rules (simple gradient ascent on weights and bias) 
        with torch.no_grad():
            layer.weight -= learning_rate * layer.weight.grad
            layer.bias   -= learning_rate * layer.bias.grad
        
    