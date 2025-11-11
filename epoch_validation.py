import torch
import torch.optim as optim
import pickle
import gc
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from scipy import stats

from test6 import NetCustom
from load_pop3 import MNISTDataLoader

def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def insert_parameters(net, individual):
    """Insert evolved parameters into the network"""
    for (layer_name, nn_dict_original), (_, nn_dict_individual) in zip(net.manager.networks.items(), individual.items()):
        for nn_name, nn_managed in nn_dict_original.items():
            individual_params = nn_dict_individual[nn_name]
            network_params = list(nn_managed.parameters())
            for param_net, param_ind in zip(network_params, individual_params):
                param_net.data = param_ind.clone().detach()

def evaluate_model(net, test_batches):
    """Evaluate model accuracy on test set"""
    net.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in test_batches:
            outputs, loss = net(images, labels)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            # Aggressive cleanup
            del outputs, predicted, loss, images, labels
    
    net.train()
    clear_memory()
    return correct / total, total_loss / len(test_batches)

def get_test_set(dataset_name, seed=42):
    """Get a fixed test set for evaluation - smaller for memory efficiency"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    fashion = (dataset_name == 'fashion_mnist')
    dataloader = MNISTDataLoader(batch_size=16, fashion=fashion)
    
    test_batches = []
    for _ in range(20):  # Reduced test set size for memory efficiency
        images, labels = dataloader.get_batch(mode='eval')
        test_batches.append((images.clone().detach(), labels.clone().detach()))
    
    clear_memory()
    return test_batches

def train_standard_model(dataset_name, num_epochs=10, seed=42, report_interval=100):
    """Train a standard model with Adam optimizer"""
    print(f"Training STANDARD model on {dataset_name} for {num_epochs} epochs...")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model and optimizer
    net = NetCustom()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)
    
    # Get data loader and test set
    fashion = (dataset_name == 'fashion_mnist')
    dataloader = MNISTDataLoader(batch_size=16, fashion=fashion)
    test_batches = get_test_set(dataset_name, seed)
    
    # Training tracking
    training_history = {
        'epoch': [],
        'batch': [],
        'loss': [],
        'accuracy': [],
        'time': []
    }
    
    net.train()
    start_time = time.time()
    batch_count = 0
    
    # Calculate batches per epoch - ultra conservative for memory efficiency
    batches_per_epoch = 500  # Fixed small number to prevent memory buildup
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_batches = 0
        
        for batch_in_epoch in range(batches_per_epoch):
            batch_count += 1
            
            # Get batch
            images, labels = dataloader.get_batch(mode='train')
            
            # Standard training step
            optimizer.zero_grad()
            outputs, loss = net(images, labels)
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_value
            epoch_batches += 1
            
            # Aggressive cleanup
            del outputs, loss, images, labels
            
            # Clear memory every 25 batches (more frequent)
            if batch_count % 25 == 0:
                clear_memory()
            
            # Report progress
            if batch_count % report_interval == 0:
                current_time = time.time() - start_time
                accuracy, test_loss = evaluate_model(net, test_batches)
                
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, "
                      f"Acc: {accuracy:.4f}, Loss: {test_loss:.4f}, "
                      f"Time: {current_time:.1f}s")
                
                training_history['epoch'].append(epoch + 1)
                training_history['batch'].append(batch_count)
                training_history['loss'].append(test_loss)
                training_history['accuracy'].append(accuracy)
                training_history['time'].append(current_time)
                
                # Limit history size to prevent memory buildup
                if len(training_history['epoch']) > 15:
                    for key in training_history:
                        training_history[key] = training_history[key][-10:]  # Keep last 10
                
                clear_memory()
        
        # End of epoch evaluation
        epoch_time = time.time() - epoch_start
        accuracy, test_loss = evaluate_model(net, test_batches)
        avg_epoch_loss = epoch_loss / epoch_batches
        
        print(f"  EPOCH {epoch+1} COMPLETE: Acc: {accuracy:.4f}, "
              f"Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Add epoch-end point
        training_history['epoch'].append(epoch + 1)
        training_history['batch'].append(batch_count)
        training_history['loss'].append(test_loss)
        training_history['accuracy'].append(accuracy)
        training_history['time'].append(time.time() - start_time)
    
    total_time = time.time() - start_time
    final_accuracy, final_loss = evaluate_model(net, test_batches)
    
    print(f"STANDARD TRAINING COMPLETE: Final Acc: {final_accuracy:.4f}, "
          f"Total Time: {total_time:.1f}s")
    
    # Save final model state for comparison
    final_state = {name: param.data.clone() for name, param in net.named_parameters()}
    
    # Aggressive cleanup
    del net, optimizer, dataloader, test_batches
    clear_memory()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return training_history, final_accuracy, final_state

def train_evolved_model(individual, dataset_name, num_epochs=10, seed=42, report_interval=100):
    """Train a model with evolved optimization rules"""
    print(f"Training EVOLVED model on {dataset_name} for {num_epochs} epochs...")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model with evolved parameters
    net = NetCustom()
    insert_parameters(net, individual)
    
    # Get data loader and test set
    fashion = (dataset_name == 'fashion_mnist')
    dataloader = MNISTDataLoader(batch_size=16, fashion=fashion)
    test_batches = get_test_set(dataset_name, seed)
    
    # Training tracking
    training_history = {
        'epoch': [],
        'batch': [],
        'loss': [],
        'accuracy': [],
        'time': []
    }
    
    net.train()
    start_time = time.time()
    batch_count = 0
    
    # Calculate batches per epoch - ultra conservative for memory efficiency
    batches_per_epoch = 500  # Fixed small number to prevent memory buildup
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_batches = 0
        
        for batch_in_epoch in range(batches_per_epoch):
            batch_count += 1
            
            # Get batch
            images, labels = dataloader.get_batch(mode='train')
            
            # Evolved training step
            net.zero_grads_g2()
            outputs, loss = net(images, labels)
            loss_value = loss.item()
            net.backprop_adv(test=False)
            
            # Apply evolved optimization rules
            adam_delta, updates = net.optimizer_step(learning_rate=0.0003)
            for p, d, u in zip(net.parameters(), adam_delta, updates):
                p.data.add_(u)  # Evolved component
                p.data.add_(d)  # Adam component (hybrid approach)
            
            epoch_loss += loss_value
            epoch_batches += 1
            
            # Aggressive cleanup
            del outputs, loss, adam_delta, updates, images, labels
            
            # Clear memory every 25 batches (more frequent)
            if batch_count % 25 == 0:
                clear_memory()
            
            # Report progress
            if batch_count % report_interval == 0:
                current_time = time.time() - start_time
                accuracy, test_loss = evaluate_model(net, test_batches)
                
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, "
                      f"Acc: {accuracy:.4f}, Loss: {test_loss:.4f}, "
                      f"Time: {current_time:.1f}s")
                
                training_history['epoch'].append(epoch + 1)
                training_history['batch'].append(batch_count)
                training_history['loss'].append(test_loss)
                training_history['accuracy'].append(accuracy)
                training_history['time'].append(current_time)
                
                # Limit history size to prevent memory buildup
                if len(training_history['epoch']) > 15:
                    for key in training_history:
                        training_history[key] = training_history[key][-10:]  # Keep last 10
                
                clear_memory()
        
        # End of epoch evaluation
        epoch_time = time.time() - epoch_start
        accuracy, test_loss = evaluate_model(net, test_batches)
        avg_epoch_loss = epoch_loss / epoch_batches
        
        print(f"  EPOCH {epoch+1} COMPLETE: Acc: {accuracy:.4f}, "
              f"Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Add epoch-end point
        training_history['epoch'].append(epoch + 1)
        training_history['batch'].append(batch_count)
        training_history['loss'].append(test_loss)
        training_history['accuracy'].append(accuracy)
        training_history['time'].append(time.time() - start_time)
    
    total_time = time.time() - start_time
    final_accuracy, final_loss = evaluate_model(net, test_batches)
    
    print(f"EVOLVED TRAINING COMPLETE: Final Acc: {final_accuracy:.4f}, "
          f"Total Time: {total_time:.1f}s")
    
    # Save final model state
    final_state = {name: param.data.clone() for name, param in net.named_parameters()}
    
    # Aggressive cleanup
    del net, dataloader, test_batches
    clear_memory()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return training_history, final_accuracy, final_state

def save_results(standard_history, evolved_history, standard_acc, evolved_acc, dataset_name):
    """Save results to CSV and create plots"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save training histories
    filename = f"epoch_validation_{dataset_name}_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'epoch', 'batch', 'loss', 'accuracy', 'time'])
        
        for i in range(len(standard_history['epoch'])):
            writer.writerow(['standard', standard_history['epoch'][i], 
                           standard_history['batch'][i], standard_history['loss'][i],
                           standard_history['accuracy'][i], standard_history['time'][i]])
        
        for i in range(len(evolved_history['epoch'])):
            writer.writerow(['evolved', evolved_history['epoch'][i], 
                           evolved_history['batch'][i], evolved_history['loss'][i],
                           evolved_history['accuracy'][i], evolved_history['time'][i]])
    
    print(f"Results saved to {filename}")
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(standard_history['time'], standard_history['accuracy'], 'b-', label='Standard', linewidth=2)
    plt.plot(evolved_history['time'], evolved_history['accuracy'], 'r-', label='Evolved', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy - {dataset_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(standard_history['time'], standard_history['loss'], 'b-', label='Standard', linewidth=2)
    plt.plot(evolved_history['time'], evolved_history['loss'], 'r-', label='Evolved', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {dataset_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Batch-based accuracy
    plt.subplot(1, 3, 3)
    plt.plot(standard_history['batch'], standard_history['accuracy'], 'b-', label='Standard', linewidth=2)
    plt.plot(evolved_history['batch'], evolved_history['accuracy'], 'r-', label='Evolved', linewidth=2)
    plt.xlabel('Training Batches')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Batches - {dataset_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"epoch_validation_plots_{dataset_name}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to {plot_filename}")

def main():
    print("EPOCH-BASED VALIDATION (Memory Optimized)")
    print("=" * 60)
    print("This will train both standard and evolved models for multiple epochs")
    print("and compare their progress over time.")
    print("ULTRA MEMORY OPTIMIZATIONS:")
    print("- Reduced batch size to 16")
    print("- Aggressive garbage collection every 25 batches") 
    print("- Fixed 500 batches per epoch")
    print("- Limited training history to last 10 points")
    print("- Reduced to 2 epochs and smaller test sets")
    print("=" * 60)
    
    # Configuration - Ultra memory optimized
    NUM_EPOCHS = 2  # Further reduced for memory efficiency
    DATASET = 'mnist'  # Change to 'fashion_mnist' for Fashion-MNIST
    SEED = 42
    REPORT_INTERVAL = 100  # Report every N batches (reduced for more frequent cleanup)
    
    print(f"Configuration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Seed: {SEED}")
    print(f"  Report interval: {REPORT_INTERVAL} batches")
    print()
    
    # Load population
    print("Loading population...")
    with open('population4.pkl', 'rb') as f:
        population = pickle.load(f)
    
    # Use individual 16 (best performer from previous validation)
    individual_16 = population[16]
    print("Using Individual 16 from population")
    print()
    
    # Train standard model
    print("PHASE 1: Training Standard Model")
    print("-" * 40)
    standard_history, standard_acc, standard_state = train_standard_model(
        DATASET, NUM_EPOCHS, SEED, REPORT_INTERVAL
    )
    print()
    
    # Train evolved model
    print("PHASE 2: Training Evolved Model")
    print("-" * 40)
    evolved_history, evolved_acc, evolved_state = train_evolved_model(
        individual_16, DATASET, NUM_EPOCHS, SEED, REPORT_INTERVAL
    )
    print()
    
    # Analysis and comparison
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"Final Accuracies:")
    print(f"  Standard Model: {standard_acc:.4f}")
    print(f"  Evolved Model:  {evolved_acc:.4f}")
    print(f"  Improvement:    {evolved_acc - standard_acc:+.4f}")
    print()
    
    # Statistical significance (rough estimate using final accuracies)
    improvement = evolved_acc - standard_acc
    improvement_pct = (improvement / standard_acc) * 100
    
    print(f"Performance Analysis:")
    if improvement > 0:
        print(f"✓ Evolved model is {improvement_pct:.2f}% better than standard")
    else:
        print(f"✗ Evolved model is {abs(improvement_pct):.2f}% worse than standard")
    
    # Convergence analysis
    if len(standard_history['accuracy']) > 5 and len(evolved_history['accuracy']) > 5:
        std_final_5 = np.mean(standard_history['accuracy'][-5:])
        evo_final_5 = np.mean(evolved_history['accuracy'][-5:])
        print(f"Convergence (last 5 measurements):")
        print(f"  Standard: {std_final_5:.4f}")
        print(f"  Evolved:  {evo_final_5:.4f}")
    
    # Save results and create plots
    print()
    print("Saving results and creating plots...")
    save_results(standard_history, evolved_history, standard_acc, evolved_acc, DATASET)
    
    print()
    print("VALIDATION COMPLETE!")
    print("Check the generated CSV file and plots for detailed analysis.")

if __name__ == "__main__":
    main()
