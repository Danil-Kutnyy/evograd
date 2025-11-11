import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomAdam:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params) # Store parameters to optimize
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {} # Dictionary to store state like m, v, and step

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        """Performs a single optimization step."""
        deltas = []
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            # Initialize state for this parameter if not already done
            if i not in self.state:
                self.state[i] = {}
                self.state[i]['step'] = 0
                # Exponential moving average of gradient values
                self.state[i]['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                self.state[i]['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
            state = self.state[i]
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.beta1, self.beta2
            state['step'] += 1
            t = state['step']
            # Weight decay
            if self.weight_decay != 0:
                grad = grad.add(p.data, alpha=self.weight_decay)
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # Bias correction
            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = self.lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
            
            # Calculate the delta
            update_delta = -step_size * (exp_avg / denom) # This is p_new - p_old

            # Apply the update using the delta
            p.data.add_(update_delta)
            deltas.append(update_delta)
        return deltas

# --- Simple Model, Data, and Training Loop ---

# You can add the simple model, data generation, and training loop here later.
# We'll do this in the next step.

# 1. Simple Model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 2. Synthetic Data Generation
def get_synthetic_data(batch_size=64, input_size=10, output_size=1):
    X = torch.randn(batch_size, input_size)
    # Simple linear relationship + noise
    true_weight = torch.randn(input_size, output_size) * 0.5
    true_bias = torch.randn(output_size) * 0.1
    y = X @ true_weight + true_bias + torch.randn(batch_size, output_size) * 0.1
    return X, y

# 3. Training Loop
def train(model, optimizer, loss_fn, X, y, steps=100):
    model.train() # Set model to training mode
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # Optional: print loss progress
        # if step % 10 == 0:
        #     print(f'Step {step}, Loss: {loss.item():.4f}')
    return losses

# --- Comparison Logic (Step 3) ---
if __name__ == '__main__':
    torch.manual_seed(42) # For reproducibility

    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 1
    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.0 # Keep it simple for initial comparison
    steps = 16

    # Data
    X, y = get_synthetic_data(batch_size=64, input_size=input_size, output_size=output_size)

    # Models (create two identical instances)
    model_custom = SimpleNet(input_size, hidden_size, output_size)
    model_torch = SimpleNet(input_size, hidden_size, output_size)
    # Copy weights to ensure they start identically
    model_torch.load_state_dict(model_custom.state_dict())

    # Optimizers
    optimizer_custom = CustomAdam(model_custom.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # Loss Function
    loss_fn = nn.MSELoss()

    # Training
    print("Training with Custom Adam...")
    losses_custom = train(model_custom, optimizer_custom, loss_fn, X, y, steps=steps)
    print("Training with PyTorch Adam...")
    losses_torch = train(model_torch, optimizer_torch, loss_fn, X, y, steps=steps)

    # Comparison
    print("\n--- Comparison Results ---")
    print(f"Final Loss (Custom): {losses_custom[-1]:.6f}")
    print(f"Final Loss (PyTorch): {losses_torch[-1]:.6f}")

    # Compare final parameters
    all_match = True
    for p_custom, p_torch in zip(model_custom.parameters(), model_torch.parameters()):
        if not torch.allclose(p_custom.data, p_torch.data, atol=1e-6):
            print(f"Parameter mismatch detected! Max difference: {torch.max(torch.abs(p_custom.data - p_torch.data))}")
            # print("Custom:", p_custom.data)
            # print("Torch:", p_torch.data)
            all_match = False
            # break # Uncomment to stop at first mismatch

    if all_match:
        print("\nSUCCESS: Final parameters match between Custom Adam and PyTorch Adam!")
    else:
        print("\nFAILURE: Final parameters DO NOT match.")
