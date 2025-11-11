import torch
import torch.optim as optim
import pickle
import sys
import gc  # Add garbage collection
from test6 import NetCustom, MNISTDataLoader

# Configuration
POPULATION_FILE = 'population4.pkl'
BATCH_SIZE = 64
NUM_TRAIN_BATCHES = 300  # Reduced further to prevent memory issues
NUM_TEST_BATCHES = 50    # Reduced 
LEARNING_RATE = 0.0003
NUM_EPOCHS = 2  # Reduced epochs for faster evaluation

def clear_memory():
    """Clear memory and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def insert_parameters(net, individual):
    """Insert evolved learning rule parameters into the network"""
    for (layer_name, nn_dict_original), (_, nn_dict_individual) in zip(net.manager.networks.items(), individual.items()):
        for nn_name, nn_managed in nn_dict_original.items():
            individual_params = nn_dict_individual[nn_name]
            network_params = list(nn_managed.parameters())
            for param_net, param_ind in zip(network_params, individual_params):
                param_net.data = param_ind.clone().detach()

def get_batches(dataloader, num_batches, mode='train'):
    """Get a fixed number of batches"""
    batches = []
    for _ in range(num_batches):
        images, labels = dataloader.get_batch(mode=mode)
        batches.append((images.clone().detach(), labels.clone().detach()))
    return batches

def test_accuracy(net, test_batches):
    """Test network accuracy on test batches"""
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_batches:
            outputs, _ = net(images, labels)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Clear intermediate results
            del outputs, predicted
    
    clear_memory()
    return correct / total

def train_standard_network(train_batches, test_batches):
    """Train a network using standard backpropagation"""
    print("=" * 60)
    print("TRAINING STANDARD NETWORK (Standard Backpropagation + Adam)")
    print("=" * 60)
    
    net = NetCustom()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        net.train()
        epoch_loss = 0
        
        for i, (images, labels) in enumerate(train_batches):
            optimizer.zero_grad()
            outputs, loss = net(images, labels)
            loss.backward()  # Standard backpropagation
            optimizer.step()  # Standard Adam optimizer
            epoch_loss += loss.item()
            
            # Clear intermediate results
            del outputs, loss
            
            if i % 50 == 0:  # Reduced frequency
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {epoch_loss/(i+1):.4f}")
                clear_memory()  # Clear memory periodically
        
        # Test accuracy
        accuracy = test_accuracy(net, test_batches)
        accuracies.append(accuracy)
        avg_loss = epoch_loss / len(train_batches)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
        clear_memory()  # Clear memory after each epoch
    
    final_accuracy = accuracies[-1]
    print(f"STANDARD NETWORK FINAL ACCURACY: {final_accuracy:.4f}")
    print()
    
    # Save state dict and clear optimizer
    state_dict = net.state_dict().copy()
    del optimizer
    
    return final_accuracy, state_dict, net

def train_evolved_network(individual, individual_idx, train_batches, test_batches, baseline_net):
    """Train a network using evolved learning rules"""
    print(f"TRAINING INDIVIDUAL #{individual_idx} (Evolved Learning Rules)")
    print("-" * 50)
    
    net = NetCustom()
    insert_parameters(net, individual)  # Insert evolved learning rule parameters
    
    # Copy weights from baseline network
    with torch.no_grad():
        for p1, p2 in zip(net.parameters(), baseline_net.parameters()):
            p1.data.copy_(p2.data)
    
    accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        net.train()
        
        for i, (images, labels) in enumerate(train_batches):
            net.zero_grads_g2()  # Zero both grad and grad2
            outputs, loss = net(images, labels)
            net.backprop_adv(test=False)  # Use evolved backpropagation
            
            # Apply evolved optimizer
            adam_delta, updates = net.optimizer_step(learning_rate=LEARNING_RATE)
            for p, d, u in zip(net.parameters(), adam_delta, updates):
                p.data.add_(u)  # Evolved updates
                p.data.add_(d)  # Standard Adam component
            
            # Clear intermediate results
            del outputs, loss, adam_delta, updates
            
            if i % 50 == 0:  # Reduced frequency
                clear_memory()  # Clear memory periodically
        
        # Test accuracy
        accuracy = test_accuracy(net, test_batches)
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1} - Test Accuracy: {accuracy:.4f}")
        
        clear_memory()  # Clear memory after each epoch
    
    final_accuracy = accuracies[-1]
    
    # Clean up network
    del net
    clear_memory()
    
    return final_accuracy

def main():
    print("Loading population...")
    try:
        with open(POPULATION_FILE, 'rb') as f:
            population = pickle.load(f)
        print(f"Loaded {len(population)} individuals from {POPULATION_FILE}")
    except FileNotFoundError:
        print(f"Error: {POPULATION_FILE} not found!")
        return
    
    print("Initializing data loader...")
    dataloader = MNISTDataLoader(batch_size=BATCH_SIZE)
    
    print(f"Preparing {NUM_TRAIN_BATCHES} training and {NUM_TEST_BATCHES} test batches...")
    train_batches = get_batches(dataloader, NUM_TRAIN_BATCHES, mode='train')
    test_batches = get_batches(dataloader, NUM_TEST_BATCHES, mode='eval')
    
    clear_memory()
    
    # First, train standard network
    baseline_accuracy, baseline_state, baseline_net = train_standard_network(train_batches, test_batches)
    
    # Store results
    results = []
    results.append(('Standard', baseline_accuracy))
    
    # Test each individual
    print("=" * 60)
    print("TESTING EVOLVED INDIVIDUALS")
    print("=" * 60)
    
    for i, individual in enumerate(population):
        try:
            evolved_accuracy = train_evolved_network(individual, i, train_batches, test_batches, baseline_net)
            results.append((f'Individual {i}', evolved_accuracy))
            
            improvement = evolved_accuracy - baseline_accuracy
            print(f"Individual {i} Final Accuracy: {evolved_accuracy:.4f} (Δ{improvement:+.4f})")
            print()
            
            # Clear individual from memory
            del individual
            clear_memory()
            
        except Exception as e:
            print(f"Error evaluating Individual {i}: {e}")
            results.append((f'Individual {i}', 0.0))
            print()
            clear_memory()
    
    # Clean up baseline network and data
    del baseline_net, train_batches, test_batches, population
    clear_memory()
    
    # Final summary
    print("=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<6} {'Name':<15} {'Accuracy':<10} {'vs Standard':<12}")
    print("-" * 50)
    
    for rank, (name, accuracy) in enumerate(results, 1):
        if name == 'Standard':
            diff_str = "baseline"
        else:
            diff = accuracy - baseline_accuracy
            diff_str = f"{diff:+.4f}"
        print(f"{rank:<6} {name:<15} {accuracy:<10.4f} {diff_str:<12}")
    
    # Find best individual
    best_evolved = max([r for r in results if r[0] != 'Standard'], key=lambda x: x[1])
    print(f"\nBest evolved individual: {best_evolved[0]} with accuracy {best_evolved[1]:.4f}")
    improvement = best_evolved[1] - baseline_accuracy
    print(f"Improvement over standard: {improvement:+.4f}")

if __name__ == "__main__":
    main() 