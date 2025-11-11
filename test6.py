import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import sys
import time

learning_rate = 3e-4

# --- Data Loader Class ---
class MNISTDataLoader:
    def __init__(self, batch_size=4, data_root='./data', train_split_ratio=0.8):
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Load the full dataset
        #full_trainset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=self.transform)
        full_trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=self.transform)
        
        # Calculate split sizes
        n_samples = len(full_trainset)
        n_train = int(n_samples * train_split_ratio)
        n_eval = n_samples - n_train
        
        # Split the dataset
        train_dataset, eval_dataset = torch.utils.data.random_split(full_trainset, [n_train, n_eval], generator=torch.Generator().manual_seed(42))
        
        # Create DataLoaders for each split
        self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.evalloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize iterators
        self._reset_iterator()

    def _reset_iterator(self):
        self.trainiter = iter(self.trainloader)
        self.evaliter = iter(self.evalloader)

    def get_batch(self, mode='train'):
        if mode == 'train':
            iterator = self.trainiter
        elif mode == 'eval':
            iterator = self.evaliter
        try:
            images, labels = next(iterator)
        except StopIteration:
            self._reset_iterator()
            if mode == 'train':
                iterator = self.trainiter
            elif mode == 'eval':
                iterator = self.evaliter
            images, labels = next(iterator)
        return images, labels
       

# --- Custom Layer Implementations ---
def statn(tensor):
    """
    Compute 10 statistics for each sub-tensor along the last dimension of an input tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of any shape (d1, d2, ..., dk, n), where n is the last dimension size.
        
    Returns:
        torch.Tensor: Output tensor of shape (d1, d2, ..., dk, 10) containing the statistics:
            [mean, std, var, median, min, max, skewness, kurtosis, entropy, mad, trimmed_mean, low, high]
            
    Raises:
        ValueError: If the tensor is empty or the last dimension size is zero, or if trimmed mean would trim too much.
    """
    # Check input validity
    if tensor.numel() == 0:
        raise ValueError("stat() requires a non-empty tensor")
    n = tensor.shape[-1]
    if n == 0:
        raise ValueError("stat() requires a non-zero last dimension")
    # Number of sub-tensors
    num_sub_tensors = tensor.numel() // n
    tensor_flat = tensor.view(num_sub_tensors, n)

    # Vectorized statistics
    mean = tensor.mean(dim=-1)
    std_dev_pop = tensor.std(dim=-1, unbiased=False)
    median_val = torch.median(tensor, dim=-1).values
    min_val = tensor.min(dim=-1).values
    max_val = tensor.max(dim=-1).values

    # Skewness and Kurtosis
    diff = tensor - mean.unsqueeze(-1)
    # Handle potential division by zero or very small std_dev
    std_dev_safe = torch.where(std_dev_pop > 1e-10, std_dev_pop, torch.ones_like(std_dev_pop))
    scaled = diff / std_dev_safe.unsqueeze(-1)
    mask = std_dev_pop > 1e-10
    skewness = torch.where(mask, scaled.pow(3).mean(dim=-1), torch.zeros_like(mean))
    kurtosis = torch.where(mask, scaled.pow(4).mean(dim=-1) - 3, torch.full_like(mean, -3.0)) # Corrected Kurtosis definition

    # Median Absolute Deviation (MAD)
    abs_dev = (tensor - median_val.unsqueeze(-1)).abs()
    mad = torch.median(abs_dev, dim=-1).values

    # Trimmed Mean (10%)
    proportiontocut = 0.1
    k = int(math.floor(n * proportiontocut))
    if 2 * k >= n and n > 0: # Added n > 0 check
        #raise ValueError("stat() trimmed mean requires more than 20 percent of elements to be trimmed")
        print(f"Warning: statn trimmed mean trimming {2*k}/{n} elements. Setting to NaN.")
        trimmed_mean = torch.full_like(mean, float('nan')) # Handle case gracefully
    elif n == 0:
         trimmed_mean = torch.full_like(mean, float('nan')) # Handle n=0 case
    else:
        sorted_tensor = torch.sort(tensor, dim=-1).values
        trimmed = sorted_tensor[..., k:-k] if k > 0 else sorted_tensor
        trimmed_mean = trimmed.mean(dim=-1)

    # Combine all statistics in the original order
    results = [
        mean,
        std_dev_pop,
        median_val,
        min_val,
        max_val,
        skewness,
        kurtosis,
        mad,
        trimmed_mean,
    ]

    # Stack along the last dimension to get shape (d1, d2, ..., dk, 10)
    output = torch.stack(results, dim=-1)

    return output

def statn_lite(tensor):
    """
    Compute 6 faster statistics for each sub-tensor along the last dimension of an input tensor.
    Excluded stats: Median, MAD, Trimmed Mean.

    Args:
        tensor (torch.Tensor): Input tensor of any shape (d1, d2, ..., dk, n).

    Returns:
        torch.Tensor: Output tensor of shape (d1, d2, ..., dk, 6) containing:
            [mean, std_dev_pop, min_val, max_val, skewness, kurtosis]
            
    Raises:
        ValueError: If the tensor is empty or the last dimension size is zero.
    """
    # Check input validity
    if tensor.numel() == 0:
        raise ValueError("statn_lite() requires a non-empty tensor")
    n = tensor.shape[-1]
    if n == 0:
        raise ValueError("statn_lite() requires a non-zero last dimension")

    # Vectorized statistics (Faster ones)
    mean = tensor.mean(dim=-1)
    std_dev_pop = tensor.std(dim=-1, unbiased=False)
    min_val = tensor.min(dim=-1).values
    max_val = tensor.max(dim=-1).values

    # Skewness and Kurtosis
    diff = tensor - mean.unsqueeze(-1)
    # Handle potential division by zero or very small std_dev
    std_dev_safe = torch.where(std_dev_pop > 1e-10, std_dev_pop, torch.ones_like(std_dev_pop))
    scaled = diff / std_dev_safe.unsqueeze(-1)
    mask = std_dev_pop > 1e-10
    skewness = torch.where(mask, scaled.pow(3).mean(dim=-1), torch.zeros_like(mean))
    kurtosis = torch.where(mask, scaled.pow(4).mean(dim=-1) - 3, torch.full_like(mean, -3.0))

    # Combine remaining statistics
    results = [
        mean,
        std_dev_pop,
        min_val,
        max_val,
        skewness,
        kurtosis,
    ]

    # Stack along the last dimension
    output = torch.stack(results, dim=-1)

    return output

def stat_stack(batch_size, *tensors):
    """
    Compute statistics for multiple tensors, ensuring all have the correct batch size,
    then broadcast intermediate dimensions and stack the results.

    This function processes a variable number of tensors by first adjusting any that
    don't match the specified batch_size. It applies the statn function to each,
    broadcasts them to a common shape based on the maximum intermediate dimensions,
    and stacks the results along a new dimension.

    Args:
        batch_size (int): The expected batch size for all tensors.
        *tensors (torch.Tensor): One or more tensors to process. Each will be adjusted
            if necessary to match the batch_size before computing statistics.

    Returns:
        torch.Tensor: A stacked tensor of statistics, shaped as (batch_size, N, ..., 10),
            where N is the number of input tensors, then flattened as needed.

    Raises:
        ValueError: If no tensors are provided or if a tensor cannot be broadcasted.
    """
    if not tensors:
        raise ValueError("At least one tensor must be provided")
    
    # Tensor Adjustment
    adjusted_tensors = []  # List to hold adjusted tensors
    for i, tensor in enumerate(tensors):
        # Check and adjust tensors that don't match the batch size
        original_shape = tensor.shape
        if tensor.shape[0] != batch_size:
            if len(tensor.shape) == 0:  # Scalar case
                adjusted = tensor.unsqueeze(0).expand(batch_size)  # Add and expand batch dimension
            elif len(tensor.shape) == 1:  # 1D tensor, e.g., (C,) - Assuming C is not the batch dim
                 # Check if it's meant to be a batch of size 1 scalars
                 if tensor.numel() == 1 and batch_size == 1:
                     adjusted = tensor.expand(batch_size) # Special case: batch of 1 scalar
                 else: # Assume it's features for a single batch item
                    adjusted = tensor.unsqueeze(0).expand(batch_size, -1) # Add and expand batch_size
            else: # Multi-dimensional, assume first dim should be batch
                 # This case might need careful handling depending on expected inputs
                 # Example adjustment (might need refinement based on use case):
                 if tensor.shape[0] == 1:
                     adjusted = tensor.expand(batch_size, *tensor.shape[1:])
                 else:
                    raise ValueError(f"Tensor {i} shape {original_shape} cannot be automatically broadcast to batch_size {batch_size}")

        else:
            adjusted = tensor  # Tensor already has the correct shape
        adjusted_tensors.append(adjusted)  # Add the adjusted tensor to the list

    # Compute Stats
    stats_list = [statn_lite(adjusted) for adjusted in adjusted_tensors]  # Compute stats for each adjusted tensor

    # Determine Target Shape
    if not stats_list: # Handle case where statn might return empty if input was weird
        # Return an empty tensor or handle appropriately
        return torch.empty((batch_size, 0)) # Example empty tensor

    target_shape = list(stats_list[0].shape)  # Base shape on the first stats tensor
    for stats in stats_list[1:]:
        # Ensure dimensions match before comparing sizes (handle different ndims)
        current_len = len(stats.shape)
        target_len = len(target_shape)
        max_len = max(current_len, target_len)

        # Pad shapes with 1s at the beginning if necessary
        current_shape_padded = [1] * (max_len - current_len) + list(stats.shape)
        target_shape_padded = [1] * (max_len - target_len) + target_shape

        new_target_shape = []
        # Compare dimensions element-wise, skip batch (dim 0) and stats (last dim)
        for dim in range(max_len):
             if dim == 0 or dim == max_len -1: # Skip batch and last dim
                 new_target_shape.append(max(current_shape_padded[dim], target_shape_padded[dim])) # Should be same usually
             else:
                 new_target_shape.append(max(current_shape_padded[dim], target_shape_padded[dim]))
        target_shape = new_target_shape # Update target shape

    # Broadcasting
    broadcasted_stats = []  # List to hold broadcasted stats tensors
    for i, stats in enumerate(stats_list):
        # Pad stats shape if necessary before broadcasting
        stats_shape_padded = [1] * (len(target_shape) - len(stats.shape)) + list(stats.shape)
        stats_expanded = stats.view(stats_shape_padded) # Reshape to match target ndim
        try:
             broadcasted = torch.broadcast_to(stats_expanded, target_shape)  # Broadcast each to the target shape
             broadcasted_stats.append(broadcasted)
        except RuntimeError as e:
             # Decide how to handle error: skip, raise, use original?
             # For now, let's re-raise to stop execution
             raise e

    # Stacking
    # Stack the broadcasted stats along a new dimension (dimension 1)
    stacked_stats = torch.stack(broadcasted_stats, dim=1)  # Results in (batch_size, N, ..., 10)

    # Flattening
    final_output = stacked_stats.view(batch_size, -1)  # Flatten as per original function

    return final_output
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
            #p.data.add_(update_delta)
            deltas.append(update_delta)
        return deltas
    
class StatPorcessingNet(nn.Module):
    def __init__(self, n=4, outputs=4):
        super().__init__()
        stats_dim = 6
        self.fc1 = nn.Linear(n*stats_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8+n*stats_dim, 8)
        self.fc3 = nn.Linear(8, outputs)
        # Initialize weights with mean=0, std=0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = torch.cat([x, inputs], dim=-1)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

class GradientProcessingNet(nn.Module):
    def __init__(self, inputs, outputs=2):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16+inputs, 4)
        self.fc3 = nn.Linear(4, outputs)
        # Initialize weights with mean=0, std=0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = torch.cat([x, inputs], dim=-1)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class GradientLiteProcessingNet(nn.Module):
    def __init__(self, inputs, outputs=2):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8+inputs, 4)
        self.fc3 = nn.Linear(4, outputs)
        # Initialize weights with mean=0, std=0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = torch.cat([x, inputs], dim=-1)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class NeuroOptimizer(nn.Module):
    def __init__(self, inputs=4, hidden_size=8):
        super().__init__()
        #input is 7, 3 for grad1, grad2, optimizer, and 4 for stats
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(inputs, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs, state):
        x, new_state = self.lstm1(inputs, state)
        x = self.relu(  x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, new_state
    
    def set_params(self, parameters):
        self.params = [p for p in parameters]
        self.states = []
        for p in self.params:
            self.states.append( [torch.zeros(1, p.data.view(-1).shape[0], self.hidden_size), torch.zeros(1, p.data.view(-1).shape[0], self.hidden_size)])

    def apply(self, deltas):
        updates = []
        for i, (param, state, delta) in enumerate(zip(self.params, self.states, deltas)):
            inp = torch.stack([param.data, param.grad, delta, param.grad2], dim=-1)
            inp = inp.view(-1, 4)
            x, new_state = self(inp.unsqueeze(1), state)
            self.states[i] = new_state
            delta2 = x.squeeze().view(param.data.shape)
            updates.append(delta2)
            '''
            update = delta + delta2
            
            try:
                param.data.add_(update)
            except Exception as e:
                print(f"Error applying update: {e}")
                print(f"Update: {update.shape}")
                print(f"Parameter: {param.data.shape}")
                print(f"Delta: {delta.shape}")
                print(f"X: {x.shape}")
                print(f"State: {state[0].shape}")
                print(f"Input: {inp.shape}")
                raise e
            '''
        return updates
class ComputationalGraph:
    def __init__(self):
        self.nodes = []  # List of nodes in topological order
        
    def add_node(self, layer, inputs):
        node = {'layer': layer, 'inputs': inputs, 'output': None}
        self.nodes.append(node)
        return node  # Return for chaining
    
    def backward(self, initial_grad1, initial_grad2):
        grad1 = initial_grad1
        grad2 = initial_grad2
        for node in reversed(self.nodes):
            grad1, grad2 = node['layer'].backprop_adv(grad1, grad2)  # Updated call

class ParameterG2(nn.Parameter):
    """
    A subclass of torch.nn.Parameter that includes an additional attribute `grad2`
    for storing a second gradient, often used in advanced optimization or analysis techniques.
    It behaves identically to nn.Parameter in all other respects.
    """
    def __new__(cls, data=None, requires_grad=True):
        # Create the parameter using the parent class's __new__
        instance = super(ParameterG2, cls).__new__(cls, data, requires_grad=requires_grad)
        # Initialize the grad2 attribute
        instance.grad2 = None
        return instance


class LearningRulesManager:
    def __init__(self):
        self.networks = {}  # {class_name: {'spn': instance, 'gpn': instance}, e.g., {'CustomLinear': {'spn': instance, 'gpn': instance}}}
    
    def build_networks(self, configurations):
        for layer, spn_config, gpn_config in configurations:
            key = layer.__class__.__name__  # Use class name as key
            self.networks[key] = {}
            if spn_config:
                class_name, n = spn_config
                self.networks[key]['spn'] = globals()[class_name](n=n)
            if gpn_config:
                class_name, inputs, outputs = gpn_config
                self.networks[key]['gpn'] = globals()[class_name](inputs=inputs, outputs=outputs)
        self.networks['Global'] = {'optimizer': NeuroOptimizer()}
    def get_networks_for_layer(self, layer):
        key = layer.__class__.__name__  # Use class name as key
        return self.networks.get(key, {'spn': None, 'gpn': None})

class GraphAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.manager = None
        self.lrs_config = None

class CustomLinear(GraphAwareModule):  # Updated inheritance
    def __init__(self, in_features, out_features):
        super().__init__()  # This will now include manager and lrs_config
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ParameterG2(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.backprop_data = {}
        self.lrs_config = {'spn': ('StatPorcessingNet', 3), 'gpn': ('GradientProcessingNet', 7, 2)}  # Keep this as is
        self.weight_grad2 = None
        # self.manager is now inherited
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
    def forward(self, input, graph=None):
        # Use F.linear for the operation
        output = F.linear(input, self.weight, None)
        if graph:
            node = graph.add_node(self, [input])  # Add to graph
            node['output'] = output
        
        self.backprop_data['input'] = input  # Store for backprop
        self.backprop_data['output'] = output
        return output

    def backprop_adv(self, grad_output1, grad_output2, manager=None):
        # Standard Gradients
        inputs = self.backprop_data['input']
        grad_input1 = grad_output1 @ self.weight  # Standard input gradient
        grad_weight1_base = grad_output1.t() @ inputs  # Standard weight gradient
        self.weight.grad = grad_weight1_base

        if grad_output2 is None:
            return grad_input1, None
        else:
            # Compute statistics similar to CustomBias
            outputs = self.backprop_data['output']
            batch_size = inputs.shape[0]  # Get batch size
            # Note: weight.view(-1) and grad_weight1_base.view(-1) can be large
            stats = stat_stack(batch_size, inputs, outputs, grad_input1)

            # SPN processing
            spn_instance = self.manager.get_networks_for_layer(self)['spn']  # Get from manager
            stats_processed = spn_instance(stats)

            # Prepare inputs for GPN
            gpn_inputs = torch.stack([outputs, grad_output1, grad_output2], dim=-1)  # Simplified inputs
            # Broadcasting stats can be expensive if shapes differ greatly
            broadcasted_stats = stats_processed.unsqueeze(1).expand(-1, gpn_inputs.shape[1], -1)
            modified_gpn_inputs = torch.cat([gpn_inputs, broadcasted_stats], dim=-1)

            # Generate two deltas from GPN
            gpn_instance = self.manager.get_networks_for_layer(self)['gpn']  # Get from manager
            grads_delta = gpn_instance(modified_gpn_inputs)  # Two values: [delta_for_weight, delta_for_input]
            
            # Apply deltas: Adjust gradients
            # Matrix multiplications here
            self.weight.grad2 = (grad_output1 + grads_delta[...,0]).t() @ inputs  # First delta for weights
            grad_input2 = (grad_output1 + grads_delta[...,1]) @ self.weight  # Second delta for inputs

            return grad_input1, grad_input2  # Return adjusted input gradients

class CustomBias(GraphAwareModule):  # Updated inheritance
    def __init__(self, features):
        super().__init__()  # This will now include manager and lrs_config
        self.bias = ParameterG2(torch.Tensor(features))
        self.reset_parameters()
        self.backprop_data = {}
        self.lrs_config = {'spn': ('StatPorcessingNet', 4), 'gpn': ('GradientProcessingNet', 9, 2)}
        self.bias_grad2 = None
        # self.manager is now inherited

    def reset_parameters(self):
        fan_in = self.bias.size(0)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input, graph=None):
        output = input + self.bias # Broadcasting handles addition
        if graph:
            node = graph.add_node(self, [input])
            node['output'] = output
        self.inputs = input
        self.outputs = output
        return output

    def backprop_adv(self, grad_output1, grad_output2, manager=None):
        # Standard Gradients
        grad_input1 = grad_output1
        # Summation can be slow for large batches/features
        self.bias.grad = grad_output1.sum(dim=0)

        if grad_output2 is None:
            return grad_input1, None
        else:
            # Compute Statistics
            inputs = self.inputs
            outputs = self.outputs
            stats = stat_stack(inputs.shape[0],inputs, self.bias, outputs, grad_input1)

            # SPN Processing
            spn_instance = self.manager.get_networks_for_layer(self)['spn']  # Get from manager
            stats = spn_instance(stats)

            # Prepare GPN Inputs
            gpn_inputs = torch.stack([inputs, self.bias.broadcast_to(inputs.shape), outputs, grad_output1, grad_output2], dim=-1)
            # Broadcasting stats can be costly
            ##broadcasted_stats = torch.broadcast_to(stats.unsqueeze(1), gpn_inputs.shape)
            broadcasted_stats = stats.unsqueeze(1).expand(-1, gpn_inputs.shape[1], -1)
            modified_gpn_inputs = torch.cat([gpn_inputs, broadcasted_stats], dim=-1)  # Concatenation cost

            # GPN Forward
            gpn_instance = self.manager.get_networks_for_layer(self)['gpn']  # Get from manager
            grads_delta = gpn_instance(modified_gpn_inputs)  # Pass the modified input to gpn

            # Apply Deltas
            self.bias.grad2 = grads_delta[...,0].sum(dim=0) # Summation cost
            grad_input2 = grads_delta[...,1] + grad_input1

            return grad_input1, grad_input2

class CustomReLU(GraphAwareModule):  # Updated inheritance
    def __init__(self):
        super().__init__()  # This will now include manager and lrs_config
        self.output_cache = None
        self.lrs_config = {'spn': ('StatPorcessingNet', 3), 'gpn': ('GradientLiteProcessingNet', 8, 1)}
        # self.manager is now inherited

    def forward(self, input, graph=None):
        # Use F.relu for the operation
        self.output_cache = F.relu(input) 
        if graph:
            node = graph.add_node(self, [input])
            node['output'] = self.output_cache
        return self.output_cache

    def backprop_adv(self, grad_output1, grad_output2, manager=None):
        # Standard Gradient
        relu_mask = (self.output_cache > 0).type_as(grad_output1)
        grad_input1 = grad_output1 * relu_mask  # Standard ReLU gradient
        
        if grad_output2 is None:
            return grad_input1, None
        else:
            # Compute statistics similar to CustomBias
            inputs = self.output_cache  # Using cached output as proxy for inputs in ReLU context
            outputs = self.output_cache  # ReLU output is the same as processed input
            stats = stat_stack(inputs.shape[0], inputs, outputs, grad_input1)  # Adjusted for ReLU

            # SPN Processing
            spn_instance = self.manager.get_networks_for_layer(self)['spn']  # Get from manager
            stats_processed = spn_instance(stats)
            
            # Prepare inputs for GPN
            gpn_inputs = torch.stack([inputs, outputs, grad_input1, grad_output2], dim=-1)  # [B, D, 3]
            
            # Broadcasting potentially large stats_processed tensor
            # Ensure shapes are compatible before broadcasting
            target_gpn_shape = gpn_inputs.shape[:-1] + (stats_processed.shape[-1],) # Target shape for stats: [B, D, stats_dim]
            
            # Check if stats_processed needs expanding (if it's [B, stats_dim])
            if len(stats_processed.shape) == len(target_gpn_shape) - 1:
                stats_processed_expanded = stats_processed.unsqueeze(1).expand(target_gpn_shape)
            elif stats_processed.shape == target_gpn_shape:
                stats_processed_expanded = stats_processed # Already correct shape
            else:
                # Attempt broadcasting, might error if incompatible shapes
                try:
                    stats_processed_expanded = torch.broadcast_to(stats_processed.unsqueeze(1), target_gpn_shape)
                except RuntimeError as e:
                    print(f"Error broadcasting stats_processed {stats_processed.shape} to target {target_gpn_shape} in CustomReLU: {e}")
                    stats_processed_expanded = torch.zeros(target_gpn_shape, device=stats_processed.device, dtype=stats_processed.dtype) # Fallback or raise
            
            modified_gpn_inputs = torch.cat([gpn_inputs, stats_processed_expanded], dim=-1) # [B, D, 3 + stats_dim]
            
            # Generate delta and adjust gradient
            gpn_instance = self.manager.get_networks_for_layer(self)['gpn']  # Get from manager
            grads_delta = gpn_instance(modified_gpn_inputs)  # Should output [B, D, 1]

            # Apply Delta
            # Ensure grads_delta is squeezed correctly if it has an extra dim of size 1
            if grads_delta.shape[-1] == 1:
                grads_delta = grads_delta.squeeze(-1) # Now [B, D]
            grad_input2 = grad_input1 + grads_delta  # Add delta [B, D] to base gradient [B, D]

            return grad_input1, grad_input2

class CustomCrossEntropyLoss(GraphAwareModule):  # Updated inheritance
    def __init__(self):
        super().__init__()  # This will now include manager and lrs_config
        self.logits_cache = None
        self.targets_cache = None
        self.lrs_config = {'spn': ('StatPorcessingNet', 5), 'gpn': ('GradientProcessingNet', 12, 1)}
        # self.manager is now inherited

    def forward(self, logits, targets, graph=None):
        self.logits_cache = logits
        self.targets_cache = targets
        # Use F.cross_entropy or manual calculation as before
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, targets, reduction='mean')
        return loss
    
    def backprop_adv(self, manager=None):
        # Restore Cache & Softmax
        logits = self.logits_cache
        targets = self.targets_cache
        
        # Restore and Fix Manual Softmax Calculation
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits - logits_max)
        exp_logits_sum = torch.sum(exp_logits, dim=1, keepdim=True)
        # Avoid division by zero in softmax
        exp_logits_sum_safe = torch.where(exp_logits_sum > 1e-10, exp_logits_sum, torch.ones_like(exp_logits_sum))
        self.softmax_cache = exp_logits / exp_logits_sum_safe # This is already the probability
        probs = self.softmax_cache # CORRECTED: Use softmax_cache directly, removed extra exp()
        
        # Standard Gradient (Logits1)
        one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).type_as(logits)
        grad_logits1 = probs - one_hot_targets
        
        # Compute statistics
        stats = stat_stack(logits.shape[0], logits, exp_logits, self.softmax_cache, probs, grad_logits1)

        # SPN Processing
        spn_instance = self.manager.get_networks_for_layer(self)['spn']  # Get from manager
        stats_processed = spn_instance(stats)

        # Compute second loss type / GPN Input Prep
        logits_max_broadcast = torch.broadcast_to(logits_max, logits.shape)
        exp_logits_sum_broadcast = torch.broadcast_to(exp_logits_sum, logits.shape)
        loss_data_stack = torch.stack([logits, logits_max_broadcast, exp_logits, exp_logits_sum_broadcast, self.softmax_cache, probs, one_hot_targets, grad_logits1], dim=-1)
        # Broadcasting stats
        stats_processed_broadcast = stats_processed.unsqueeze(1).expand(-1, loss_data_stack.shape[1], -1)
        modified_gpn_inputs = torch.cat([loss_data_stack, stats_processed_broadcast], dim=-1)

        # GPN Forward
        gpn_instance = self.manager.get_networks_for_layer(self)['gpn']  # Get from manager
        # Ensure output is squeezed if necessary
        gpn_output = gpn_instance(modified_gpn_inputs) # Expected shape [B, NumClasses, 1]
        if gpn_output.shape[-1] == 1:
            loss_delta = gpn_output.squeeze(-1) # Shape [B, NumClasses]
        else:
            # Handle unexpected shape, maybe error or log
            print(f"Warning: Unexpected GPN output shape {gpn_output.shape} in CrossEntropyLoss")
            loss_delta = gpn_output[..., 0] # Assume first element is the delta

        # Calculate GradLogits2
        grad_logits2 = grad_logits1 + loss_delta

        '''
        # Recalculate grad_logits1 from scratch using only logits and targets
        softmax_probs_from_scratch = F.softmax(logits, dim=1)  # Compute softmax directly from logits
        one_hot_targets_from_scratch = F.one_hot(targets, num_classes=logits.shape[1]).type_as(logits)  # Compute one-hot directly from targets
        grad_logits1_from_scratch = softmax_probs_from_scratch - one_hot_targets_from_scratch  # Standard gradient from scratch
        grad_logits1 = grad_logits1_from_scratch  # Overwrite with this from-scratch calculation
        '''
        # Mean reduction
        grad_logits1 /= logits.shape[0]
        grad_logits2 /= logits.shape[0]

        return grad_logits1, grad_logits2

# --- Original PyTorch Model ---

class NetOriginal(nn.Module):
    def __init__(self):
        super(NetOriginal, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        
        loss = None
        if y is not None:
            loss = self.loss_function(output, y)
            
        return output, loss

# --- Base Class for EvoGrad Networks ---

class EvoGradNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = ComputationalGraph()
        self.manager = LearningRulesManager()
        self.loss_function = None # To be set by subclass
        self.adam = None
    def _setup_evograd(self):
        """
        Initializes the LearningRulesManager and assigns it to relevant layers
        after the subclass has defined its layers and loss function.
        """
        if self.loss_function is None:
            raise ValueError("loss_function must be set before calling _setup_evograd.")

        # Collect configurations including the loss layer
        configurations = []
        modules_to_configure = list(self.children()) + [self.loss_function]

        for module in modules_to_configure:
            # Check direct attributes and recursively check submodules if it's a container
            items_to_check = []
            if hasattr(module, 'lrs_config'):
                 items_to_check.append(module)
            if isinstance(module, nn.Module): # Include submodules if any
                 items_to_check.extend(list(module.modules()))

            for item in items_to_check:
                 if hasattr(item, 'lrs_config') and item not in [m[0] for m in configurations]: # Avoid duplicates
                     lrs_config = getattr(item, 'lrs_config', {})
                     spn_cfg = lrs_config.get('spn')
                     gpn_cfg = lrs_config.get('gpn')
                     # Ensure we only add layers that have at least one config
                     if spn_cfg or gpn_cfg:
                        configurations.append((item, spn_cfg, gpn_cfg))


        self.manager.build_networks(configurations)

        # Assign manager to layers that need it (including the loss layer)
        all_modules = list(self.modules()) + [self.loss_function]
        for module in all_modules:
             # Check if the module itself needs a manager
             if hasattr(module, 'manager') and getattr(module, 'manager', 1) is None: # Check if manager is None
                 module.manager = self.manager
             # Recursively check submodules (e.g., within Sequential) - though less common with current structure
             if isinstance(module, nn.Module):
                 for sub_module in module.modules():
                     if hasattr(sub_module, 'manager') and getattr(sub_module, 'manager', 1) is None:
                         sub_module.manager = self.manager


    def forward(self, x, y=None):
        # Subclasses MUST implement their specific forward pass logic
        raise NotImplementedError("Subclasses must implement the forward method.")

    def backprop_adv(self, test=False):
        
        """
        Performs the advanced backpropagation using the stored graph and loss function.
        """
        if self.loss_function is None:
            raise RuntimeError("Loss function not set. Call _setup_evograd in subclass __init__.")
        if not hasattr(self.loss_function, 'backprop_adv'):
             raise AttributeError("The assigned loss_function does not have a 'backprop_adv' method.")
        if not self.graph.nodes:
             print("Warning: Computational graph is empty. Did you run forward pass with graph enabled?")
             return

        # Start the custom backward pass from the loss function
        initial_grad1, initial_grad2 = self.loss_function.backprop_adv(manager=self.manager) # Pass manager

        if test:
            initial_grad2 = None

        # Propagate gradients through the graph (layers use their assigned manager)
        self.graph.backward(initial_grad1, initial_grad2)
    
    def optimizer_step(self, learning_rate=learning_rate,):
        optimizer = self.manager.networks['Global']['optimizer']
        if self.adam is None:
            self.adam = CustomAdam(self.parameters(), lr=learning_rate)
            optimizer.set_params(self.parameters())
        
        deltas = self.adam.step()
        updates = optimizer.apply(deltas)
        return updates, deltas


    def zero_grads_g2(self):
         """Helper to zero out both grad and grad2 for ParameterG2 instances."""
         for p in self.parameters():
             if p.grad is not None:
                 p.grad.detach_()
                 p.grad.zero_()
             if isinstance(p, ParameterG2) and p.grad2 is not None:
                 p.grad2.detach_()
                 p.grad2.zero_()

# --- Custom Model using Custom Layers inheriting from EvoGradNet ---
class NetCustom(EvoGradNet): # Inherit from EvoGradNet
    def __init__(self):
        super().__init__() # Call EvoGradNet's __init__

        # Define layers as before
        self.fc1 = CustomLinear(28 * 28, 128)
        self.bias1 = CustomBias(128)
        self.relu = CustomReLU()
        self.fc2 = CustomLinear(128, 10)
        self.bias2 = CustomBias(10)

        # Define the loss function instance
        self.loss_function = CustomCrossEntropyLoss()

        # Setup manager and loss function using the base class method
        # This MUST be called AFTER all layers (including loss) are defined
        self._setup_evograd()

    # The forward method now implicitly uses nn.Module.__call__
    # and needs to build the graph specific to this architecture
    def forward(self, x, y=None):
        self.graph = ComputationalGraph()  # Reset graph per forward pass
        x = x.view(-1, 28 * 28)

        # Forward pass through layers using their forward methods
        # Pass the graph to each layer that should be part of backprop_adv
        x = self.fc1(x, self.graph)
        x = self.bias1(x, self.graph)
        x = self.relu(x, self.graph)
        x = self.fc2(x, self.graph)
        output = self.bias2(x, self.graph)

        loss = None
        if y is not None and self.loss_function is not None:
            # Calculate loss using the custom loss function's forward
            # Pass the graph so the loss function can be added to it if needed (depends on its design)
            # Our current CustomCrossEntropyLoss doesn't add itself, but it's good practice
            loss = self.loss_function(output, y, self.graph)

        return output, loss

    # backprop_adv is now inherited from EvoGradNet, no need to redefine here
    # zero_grads_g2 is also inherited
class NetBuilder(EvoGradNet): # Inherit from EvoGradNet
    def __init__(self, sizes=[128, 10]):
        super().__init__() # Call EvoGradNet's __init__
    
        input_size = 28 * 28
        layers = []
        for i, output in enumerate(sizes):
            fc = CustomLinear(input_size, output)
            bias = CustomBias(output)
            if i < len(sizes) - 1:
                relu = CustomReLU()
                layers.extend([fc, bias, relu])
            else:
                layers.extend([fc, bias])
            input_size = output
            
        self.layers = nn.ModuleList(layers)
        # Define the loss function instance
        self.loss_function = CustomCrossEntropyLoss()

        # Setup manager and loss function using the base class method
        # This MUST be called AFTER all layers (including loss) are defined
        self._setup_evograd()

    # The forward method now implicitly uses nn.Module.__call__
    # and needs to build the graph specific to this architecture
    def forward(self, x, y=None):
        self.graph = ComputationalGraph()  # Reset graph per forward pass
        x = x.view(-1, 28 * 28)

        for layer in self.layers:
            x = layer(x, self.graph)
        output = x
        
        loss = None
        if y is not None and self.loss_function is not None:
            loss = self.loss_function(output, y, self.graph)
        return output, loss


# --- Main Execution Logic (Adjustments) ---

if __name__ == "__main__":
    torch.manual_seed(42)

    # Data Loading using the new class
    dataloader = MNISTDataLoader(batch_size=4)
    images, labels = dataloader.get_batch()

    # Model Initialization
    net_original = NetOriginal()
    #net_custom = NetCustom() # Uses the new structure
    net_custom = NetCustom()
    # Zero gradients before backward passes
    net_original.zero_grad()
    net_custom.zero_grads_g2() # Use the new helper method

    # Parameter copying (should still work as layers are nn.Modules)

    #isner parameter from net_original to net_custom
    with torch.no_grad():
        for param, param_ind in zip(net_custom.parameters(), net_original.parameters()):
            param.data = param_ind.data.clone().detach()
    
    # Original Model: Forward and Backward Pass
    outputs_orig, loss_orig = net_original(images, labels)
    print(f"Original Loss: {loss_orig.item():.6f}")
    loss_orig.backward() # Standard PyTorch backward

    # Custom Model: Forward and Custom Backward Pass
    outputs_custom, loss_custom = net_custom(images, labels)
    print(f"Custom Loss: {loss_custom.item():.6f}")
    # Run the custom backward pass (method inherited from EvoGradNet)
    net_custom.backprop_adv(test=False)

    # Loss Comparison
    if torch.allclose(loss_orig, loss_custom, atol=1e-6) and torch.allclose(outputs_orig, outputs_custom, atol=1e-6):
        print("SUCCESS: Outputs match and Losses match!")
    elif not torch.allclose(loss_orig, loss_custom, atol=1e-6):
        diff = torch.abs(loss_orig - loss_custom).item()
        print(f"FAILURE: Losses do not match! Abs diff: {diff:.4e}")
    elif not torch.allclose(outputs_orig, outputs_custom, atol=1e-6):
        diff = torch.abs(outputs_orig - outputs_custom).mean().item() # Added mean for better reporting
        print(f"FAILURE: Outputs do not match! Abs diff mean: {diff:.4e}")

    # Gradient Comparison
    # Access parameters using the standard .parameters() method from nn.Module
    orig_params = list(net_original.parameters())
    cust_params = list(net_custom.parameters())
    all_close = True

    #print("Comparing Gradients:")
    param_names_orig = [name for name, _ in net_original.named_parameters()]
    param_names_cust = [name for name, _ in net_custom.named_parameters()]
    #print('param_names_orig:', len(param_names_orig), 'param_names_cust:', len(param_names_cust))
    # Ensure the parameter lists align correctly
    if len(orig_params) != len(cust_params):
         print("FAILURE: Parameter count mismatch!")
         all_close = False
    else:
        for i, (orig, cust) in enumerate(zip(orig_params, cust_params)):
            orig_name = param_names_orig[i]
            cust_name = param_names_cust[i] # Assumes order is the same

            #print(f"Comparing {orig_name} vs {cust_name}")
            if orig.grad is None and cust.grad is None:
                 #print("  (Both grads are None)")
                 continue
            if orig.grad is None or cust.grad is None:
                 print(f"  FAILURE: One grad is None (orig: {orig.grad is None}, cust: {cust.grad is None})")
                 all_close = False
                 continue

            if not torch.allclose(orig.grad, cust.grad, atol=1e-6):
                # Check if the mismatch is exactly by a factor of 2
                #if torch.allclose(orig.grad * 2, cust.grad, atol=1e-6):
                #    print(f'  NOTE: Gradients for {cust_name} mismatch by factor of 2.')
                #else:
                grad_diff = torch.abs(orig.grad - cust.grad).mean().item()
                print(f"  FAILURE: Gradients mismatch for {cust_name}! Mean Abs Diff: {grad_diff:.4e}")
                # print("Original grad:", orig.grad) # Optional: print grads for debugging
                # print("Custom grad:", cust.grad)
                all_close = False
            #else:
                #print("  SUCCESS: Gradients match.")

    if all_close:
        print("Overall SUCCESS: Gradients match!")
    else:
        print("Overall FAILURE: Gradients do not match!") 
            

    # --- Test NetBuilder ---

    net_builder = NetBuilder(sizes=[1024, 256, 128, 10])
    net_builder_optimizer = CustomAdam(net_builder.parameters(), lr=learning_rate)
    outputs_builder, loss_builder = net_builder(images, labels)

    with torch.no_grad():
        net_builder.backprop_adv(test=True)
    params = []
    adam_applied = []
    for param in net_builder.parameters():
        params.append(param.grad.data.clone().detach())
    #net_builder.optimizer_step()
    deltas = net_builder_optimizer.step()
    for param in net_builder.parameters():
        adam_applied.append(param.data.clone().detach())    
        param.grad = None
        
    adam_optimizer = torch.optim.Adam(net_builder.parameters(), lr=learning_rate)

    outputs_builder, loss_builder = net_builder(images, labels)
    loss_builder.backward()

    for param_real, param_my in zip(net_builder.parameters(), params):
        if torch.allclose(param_real.grad, param_my, atol=1e-6):
            print("SUCCESS: Gradients match!")
        else:
            print("FAILURE: Gradients do not match!")

    adam_optimizer.step()
    for param_real, param_my in zip(net_builder.parameters(), adam_applied):
        if torch.allclose(param_real.data, param_my, atol=1e-6):
            print("SUCCESS: Parameters match!")
        else:
            print("Overall FAILURE: Direct CustomAdam step does NOT match torch.optim.Adam step! (Sanity Check)")


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
        if type(outputs) == tuple:
            outputs = outputs[0]
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
    input_size = 28*28
    hidden_size = 32
    output_size = 10
    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.0 # Keep it simple for initial comparison
    steps = 16

    # Data
    X, y = get_synthetic_data(batch_size=64, input_size=input_size, output_size=output_size)

    # Models (create two identical instances)
    model_custom = NetBuilder([input_size, hidden_size, output_size])
    model_torch = SimpleNet(input_size, hidden_size, output_size)
    # Copy weights to ensure they start identically
    #model_torch.load_state_dict(model_custom.state_dict())
    for param, custom_param in zip(model_torch.parameters(), model_custom.parameters()):
        param.data = custom_param.data.clone().detach()
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
