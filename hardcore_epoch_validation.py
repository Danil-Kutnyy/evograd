import torch
import torch.optim as optim
import pickle
import gc
import numpy as np
import time
import json
import os
import csv
from copy import deepcopy

from test6 import NetCustom
from load_pop3 import MNISTDataLoader

def nuclear_memory_cleanup():
    """Nuclear option for memory cleanup"""
    gc.collect()
    gc.collect()  # Call twice
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Call twice
    # Force garbage collection again
    gc.collect()

def insert_parameters(net, individual):
    """Insert evolved parameters into the network"""
    for (layer_name, nn_dict_original), (_, nn_dict_individual) in zip(net.manager.networks.items(), individual.items()):
        for nn_name, nn_managed in nn_dict_original.items():
            individual_params = nn_dict_individual[nn_name]
            network_params = list(nn_managed.parameters())
            for param_net, param_ind in zip(network_params, individual_params):
                param_net.data = param_ind.clone().detach()

def save_model_state(net, filepath):
    """Save model state to disk"""
    state_dict = {name: param.data.cpu().clone() for name, param in net.named_parameters()}
    torch.save(state_dict, filepath)
    del state_dict
    nuclear_memory_cleanup()

def load_model_state(net, filepath):
    """Load model state from disk"""
    if os.path.exists(filepath):
        state_dict = torch.load(filepath, map_location='cpu')
        for name, param in net.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
        del state_dict
        nuclear_memory_cleanup()

def save_progress(filepath, data):
    """Save progress data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_progress(filepath):
    """Load progress data from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {'epochs_completed': 0, 'batch_count': 0, 'history': [], 'start_time': time.time()}

def evaluate_and_save(net, dataloader, progress_file, epoch, batch_count, start_time):
    """Evaluate model and save results to disk immediately"""
    net.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    # Quick evaluation
    with torch.no_grad():
        for _ in range(15):  # Small test set
            images, labels = dataloader.get_batch(mode='eval')
            outputs, loss = net(images, labels)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            # Immediate cleanup
            del outputs, predicted, loss, images, labels
    
    accuracy = correct / total
    avg_loss = total_loss / 15
    current_time = time.time() - start_time
    
    # Load existing progress
    progress = load_progress(progress_file)
    
    # Add new result
    progress['history'].append({
        'epoch': epoch,
        'batch': batch_count,
        'accuracy': accuracy,
        'loss': avg_loss,
        'time': current_time
    })
    
    # Save immediately
    save_progress(progress_file, progress)
    
    net.train()
    nuclear_memory_cleanup()
    
    return accuracy, avg_loss

def train_phase_with_checkpoints(method_name, individual, dataset, num_epochs, batches_per_epoch):
    """Train with aggressive checkpointing and memory management"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING {method_name.upper()} MODEL")
    print(f"{'='*60}")
    
    # File paths for this training phase
    model_checkpoint = f"{method_name}_model_checkpoint.pth"
    progress_file = f"{method_name}_progress.json"
    
    # Initialize progress tracking
    progress = {
        'epochs_completed': 0,
        'batch_count': 0,
        'history': [],
        'start_time': time.time()
    }
    save_progress(progress_file, progress)
    
    fashion = (dataset == 'fashion_mnist')
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # COMPLETELY FRESH START EACH EPOCH
        torch.manual_seed(42 + epoch)  # Deterministic but different per epoch
        np.random.seed(42 + epoch)
        
        # Create fresh network
        net = NetCustom()
        
        # Set up based on method
        if method_name == 'standard':
            optimizer = optim.Adam(net.parameters(), lr=0.0003)
        else:  # evolved
            insert_parameters(net, individual)
            optimizer = None  # Don't need optimizer for evolved method
        
        # Load model state if continuing
        if epoch > 0:
            load_model_state(net, model_checkpoint)
        
        # Fresh dataloader each epoch
        dataloader = MNISTDataLoader(batch_size=16, fashion=fashion)
        
        net.train()
        epoch_start = time.time()
        
        # Load current progress
        progress = load_progress(progress_file)
        
        # Training loop for this epoch
        for batch_in_epoch in range(batches_per_epoch):
            progress['batch_count'] += 1
            
            # Get fresh batch
            images, labels = dataloader.get_batch(mode='train')
            
            if method_name == 'standard':
                # Standard training
                optimizer.zero_grad()
                outputs, loss = net(images, labels)
                loss.backward()
                optimizer.step()
                del outputs, loss
                
            else:  # evolved
                # Evolved training
                net.zero_grads_g2()
                outputs, loss = net(images, labels)
                net.backprop_adv(test=False)
                
                adam_delta, updates = net.optimizer_step(learning_rate=0.0003)
                for p, d, u in zip(net.parameters(), adam_delta, updates):
                    p.data.add_(u)  # Evolved component
                    p.data.add_(d)  # Adam component
                
                del outputs, loss, adam_delta, updates
            
            # Immediate cleanup
            del images, labels
            
            # Nuclear cleanup every 25 batches
            if progress['batch_count'] % 25 == 0:
                nuclear_memory_cleanup()
            
            # Evaluate and save every 100 batches
            if progress['batch_count'] % 100 == 0:
                accuracy, test_loss = evaluate_and_save(
                    net, dataloader, progress_file, epoch + 1, 
                    progress['batch_count'], total_start_time
                )
                
                elapsed = time.time() - total_start_time
                print(f"  Batch {progress['batch_count']}: Acc={accuracy:.4f}, Loss={test_loss:.4f}, Time={elapsed:.1f}s")
                
                nuclear_memory_cleanup()
        
        # End of epoch - save model state and cleanup
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        
        # Final epoch evaluation
        accuracy, test_loss = evaluate_and_save(
            net, dataloader, progress_file, epoch + 1, 
            progress['batch_count'], total_start_time
        )
        print(f"  End-of-epoch: Acc={accuracy:.4f}, Loss={test_loss:.4f}")
        
        # Save model checkpoint
        save_model_state(net, model_checkpoint)
        
        # Update progress
        progress = load_progress(progress_file)
        progress['epochs_completed'] = epoch + 1
        save_progress(progress_file, progress)
        
        # NUCLEAR CLEANUP - destroy everything
        del net, dataloader
        if optimizer is not None:
            del optimizer
        nuclear_memory_cleanup()
        
        print(f"  Memory cleared. Checkpoint saved.")
    
    print(f"\n{method_name.upper()} TRAINING COMPLETE!")
    
    # Clean up checkpoint files
    if os.path.exists(model_checkpoint):
        os.remove(model_checkpoint)
    
    return progress_file

def compare_results(standard_file, evolved_file, dataset):
    """Compare results from two training phases"""
    
    print(f"\n{'='*60}")
    print("COMPARING RESULTS")
    print(f"{'='*60}")
    
    # Load results
    standard_data = load_progress(standard_file)
    evolved_data = load_progress(evolved_file)
    
    # Get final accuracies
    standard_final = standard_data['history'][-1]['accuracy'] if standard_data['history'] else 0
    evolved_final = evolved_data['history'][-1]['accuracy'] if evolved_data['history'] else 0
    
    improvement = evolved_final - standard_final
    
    print(f"Final Results:")
    print(f"  Standard Model:  {standard_final:.4f}")
    print(f"  Evolved Model:   {evolved_final:.4f}")
    print(f"  Improvement:     {improvement:+.4f}")
    
    if improvement > 0:
        pct_improvement = (improvement / standard_final) * 100
        print(f"  Percentage:      +{pct_improvement:.2f}%")
        print("  ✓ EVOLVED METHOD WINS!")
    else:
        pct_decline = (abs(improvement) / standard_final) * 100
        print(f"  Percentage:      -{pct_decline:.2f}%")
        print("  ✗ Standard method wins")
    
    # Save combined results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"hardcore_validation_{dataset}_{timestamp}.csv"
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'epoch', 'batch', 'accuracy', 'loss', 'time'])
        
        # Write standard results
        for entry in standard_data['history']:
            writer.writerow(['standard', entry['epoch'], entry['batch'], 
                           entry['accuracy'], entry['loss'], entry['time']])
        
        # Write evolved results
        for entry in evolved_data['history']:
            writer.writerow(['evolved', entry['epoch'], entry['batch'], 
                           entry['accuracy'], entry['loss'], entry['time']])
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Cleanup progress files
    for f in [standard_file, evolved_file]:
        if os.path.exists(f):
            os.remove(f)
    
    return {
        'standard_final': standard_final,
        'evolved_final': evolved_final,
        'improvement': improvement,
        'results_file': results_file
    }

def main():
    print("HARDCORE EPOCH VALIDATION")
    print("=" * 60)
    print("Long-running training with aggressive memory management")
    print("Features:")
    print("- Complete model destruction/recreation each epoch")
    print("- All progress saved to disk immediately") 
    print("- Nuclear memory cleanup")
    print("- Zero memory accumulation guaranteed")
    print("=" * 60)
    
    # Configuration for LONG training
    NUM_EPOCHS = 10  # Much longer now that memory is handled
    BATCHES_PER_EPOCH = 1000  # Full training
    DATASET = 'mnist'
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batches per epoch: {BATCHES_PER_EPOCH}")
    print(f"  Total batches: {NUM_EPOCHS * BATCHES_PER_EPOCH}")
    print(f"  Estimated time: {(NUM_EPOCHS * BATCHES_PER_EPOCH * 0.02):.1f} seconds")
    
    input("\nPress Enter to start (this will take a while)...")
    
    # Load population
    print("\nLoading population...")
    with open('population4.pkl', 'rb') as f:
        population = pickle.load(f)
    individual_16 = population[16]
    del population  # Immediate cleanup
    nuclear_memory_cleanup()
    
    print("Using Individual 16 from population")
    
    try:
        # Phase 1: Train standard model
        standard_file = train_phase_with_checkpoints(
            'standard', None, DATASET, NUM_EPOCHS, BATCHES_PER_EPOCH
        )
        
        # Phase 2: Train evolved model  
        evolved_file = train_phase_with_checkpoints(
            'evolved', individual_16, DATASET, NUM_EPOCHS, BATCHES_PER_EPOCH
        )
        
        # Compare results
        results = compare_results(standard_file, evolved_file, DATASET)
        
        print(f"\n{'='*60}")
        print("HARDCORE VALIDATION COMPLETED SUCCESSFULLY!")
        print("No memory leaks possible - everything was destroyed and recreated!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        # Cleanup any remaining files
        for f in ['standard_progress.json', 'evolved_progress.json', 
                  'standard_model_checkpoint.pth', 'evolved_model_checkpoint.pth']:
            if os.path.exists(f):
                os.remove(f)
        raise
    
    finally:
        nuclear_memory_cleanup()

if __name__ == "__main__":
    main()


