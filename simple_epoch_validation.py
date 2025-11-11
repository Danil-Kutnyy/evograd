import torch
import torch.optim as optim
import pickle
import gc
import numpy as np
import time
import csv

from test6 import NetCustom
from load_pop3 import MNISTDataLoader

def clear_memory():
    """Ultra aggressive memory cleanup"""
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

def evaluate_accuracy(net, dataloader, num_batches=10):
    """Quick accuracy evaluation with minimal memory use"""
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            images, labels = dataloader.get_batch(mode='eval')
            outputs, _ = net(images, labels)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del outputs, predicted, images, labels
    
    net.train()
    clear_memory()
    return correct / total

def simple_train_and_compare(individual, dataset='mnist', batches_to_train=1000):
    """Simple training comparison with minimal memory footprint"""
    print(f"\nSIMPLE TRAINING COMPARISON")
    print("=" * 50)
    print(f"Dataset: {dataset}")
    print(f"Batches to train: {batches_to_train}")
    print("=" * 50)
    
    # Setup
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    fashion = (dataset == 'fashion_mnist')
    dataloader = MNISTDataLoader(batch_size=16, fashion=fashion)
    
    results = {
        'standard': {'batches': [], 'accuracy': []},
        'evolved': {'batches': [], 'accuracy': []}
    }
    
    # PHASE 1: Standard Training
    print("\nPHASE 1: Standard Training")
    print("-" * 30)
    
    torch.manual_seed(seed)
    net_standard = NetCustom()
    optimizer = optim.Adam(net_standard.parameters(), lr=0.0003)
    
    net_standard.train()
    start_time = time.time()
    
    for batch_idx in range(batches_to_train):
        # Get batch
        images, labels = dataloader.get_batch(mode='train')
        
        # Standard training step
        optimizer.zero_grad()
        outputs, loss = net_standard(images, labels)
        loss.backward()
        optimizer.step()
        
        # Cleanup
        del outputs, loss, images, labels
        
        # Report every 200 batches
        if (batch_idx + 1) % 200 == 0:
            accuracy = evaluate_accuracy(net_standard, dataloader)
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}: Acc = {accuracy:.4f}, Time = {elapsed:.1f}s")
            
            results['standard']['batches'].append(batch_idx + 1)
            results['standard']['accuracy'].append(accuracy)
            
            clear_memory()
    
    # Final evaluation
    final_standard_acc = evaluate_accuracy(net_standard, dataloader, 20)
    print(f"  FINAL Standard Accuracy: {final_standard_acc:.4f}")
    
    # Save baseline weights
    baseline_state = {name: param.data.clone() for name, param in net_standard.named_parameters()}
    
    # Cleanup standard model
    del net_standard, optimizer
    clear_memory()
    
    # PHASE 2: Evolved Training
    print("\nPHASE 2: Evolved Training")
    print("-" * 30)
    
    torch.manual_seed(seed)
    net_evolved = NetCustom()
    insert_parameters(net_evolved, individual)
    
    # Start from same baseline weights
    for name, param in net_evolved.named_parameters():
        param.data.copy_(baseline_state[name])
    
    net_evolved.train()
    start_time = time.time()
    
    for batch_idx in range(batches_to_train):
        # Get batch
        images, labels = dataloader.get_batch(mode='train')
        
        # Evolved training step
        net_evolved.zero_grads_g2()
        outputs, loss = net_evolved(images, labels)
        net_evolved.backprop_adv(test=False)
        
        # Apply evolved optimization
        adam_delta, updates = net_evolved.optimizer_step(learning_rate=0.0003)
        for p, d, u in zip(net_evolved.parameters(), adam_delta, updates):
            p.data.add_(u)  # Evolved component
            p.data.add_(d)  # Adam component
        
        # Cleanup
        del outputs, loss, adam_delta, updates, images, labels
        
        # Report every 200 batches
        if (batch_idx + 1) % 200 == 0:
            accuracy = evaluate_accuracy(net_evolved, dataloader)
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}: Acc = {accuracy:.4f}, Time = {elapsed:.1f}s")
            
            results['evolved']['batches'].append(batch_idx + 1)
            results['evolved']['accuracy'].append(accuracy)
            
            clear_memory()
    
    # Final evaluation
    final_evolved_acc = evaluate_accuracy(net_evolved, dataloader, 20)
    print(f"  FINAL Evolved Accuracy: {final_evolved_acc:.4f}")
    
    # Cleanup evolved model
    del net_evolved, baseline_state, dataloader
    clear_memory()
    
    # Comparison
    print("\nCOMPARISON RESULTS")
    print("=" * 50)
    print(f"Standard Final Accuracy:  {final_standard_acc:.4f}")
    print(f"Evolved Final Accuracy:   {final_evolved_acc:.4f}")
    improvement = final_evolved_acc - final_standard_acc
    print(f"Improvement:              {improvement:+.4f}")
    
    if improvement > 0:
        pct_improvement = (improvement / final_standard_acc) * 100
        print(f"Percentage Improvement:   {pct_improvement:.2f}%")
        print("✓ Evolved method is better!")
    else:
        pct_decline = (abs(improvement) / final_standard_acc) * 100
        print(f"Percentage Decline:       {pct_decline:.2f}%")
        print("✗ Standard method is better")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"simple_validation_{dataset}_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'batch', 'accuracy'])
        
        for i, batch in enumerate(results['standard']['batches']):
            writer.writerow(['standard', batch, results['standard']['accuracy'][i]])
        
        for i, batch in enumerate(results['evolved']['batches']):
            writer.writerow(['evolved', batch, results['evolved']['accuracy'][i]])
        
        writer.writerow(['final', 'standard', final_standard_acc])
        writer.writerow(['final', 'evolved', final_evolved_acc])
    
    print(f"\nResults saved to: {filename}")
    
    return {
        'standard_final': final_standard_acc,
        'evolved_final': final_evolved_acc,
        'improvement': improvement,
        'results': results
    }

def main():
    print("SIMPLE EPOCH VALIDATION")
    print("=" * 60)
    print("Ultra-lightweight comparison between standard and evolved training")
    print("Designed to avoid memory issues with minimal resource usage")
    print("=" * 60)
    
    # Load population
    print("Loading population...")
    with open('population4.pkl', 'rb') as f:
        population = pickle.load(f)
    
    # Use individual 16
    individual_16 = population[16]
    print("Using Individual 16 from population")
    
    # Configuration
    BATCHES_TO_TRAIN = 800  # Conservative number
    DATASET = 'mnist'
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Batches to train: {BATCHES_TO_TRAIN}")
    print(f"  Batch size: 16")
    print(f"  Memory cleanup: Every batch")
    
    # Run comparison
    try:
        results = simple_train_and_compare(individual_16, DATASET, BATCHES_TO_TRAIN)
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Try reducing BATCHES_TO_TRAIN further if memory issues persist")
    
    finally:
        clear_memory()

if __name__ == "__main__":
    main()


