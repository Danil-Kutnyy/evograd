import torch
import torch.optim as optim
import pickle
import gc
import numpy as np
from scipy import stats

from test6 import NetCustom
from load_pop3 import MNISTDataLoader

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def insert_parameters(net, individual):
    for (layer_name, nn_dict_original), (_, nn_dict_individual) in zip(net.manager.networks.items(), individual.items()):
        for nn_name, nn_managed in nn_dict_original.items():
            individual_params = nn_dict_individual[nn_name]
            network_params = list(nn_managed.parameters())
            for param_net, param_ind in zip(network_params, individual_params):
                param_net.data = param_ind.clone().detach()

def get_batches(dataset_name, seed, num_batches=2000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    fashion = (dataset_name == 'fashion_mnist')
    dataloader = MNISTDataLoader(batch_size=64, fashion=fashion)
    
    train_batches = []
    test_batches = []
    
    for _ in range(num_batches):
        images, labels = dataloader.get_batch(mode='train')
        train_batches.append((images.clone().detach(), labels.clone().detach()))
    
    for _ in range(20):  # Smaller test set
        images, labels = dataloader.get_batch(mode='eval')
        test_batches.append((images.clone().detach(), labels.clone().detach()))
    
    return train_batches, test_batches

def test_accuracy(net, test_batches):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_batches:
            outputs, _ = net(images, labels)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del outputs, predicted
    
    clear_memory()
    return correct / total

def test_methods(individual, dataset='mnist', seeds=[42]):
    """Test all three methods with multiple seeds"""
    
    print(f"Testing on {dataset} with {len(seeds)} seeds...")
    
    standard_results = []
    evolved_only_results = []
    evolved_hybrid_results = []
    
    for seed in seeds:
        print(f"  Seed {seed}...")
        
        # Get data
        train_batches, test_batches = get_batches(dataset, seed)
        
        # Method 1: Standard (backprop + Adam only)
        torch.manual_seed(seed)
        net_standard = NetCustom()
        optimizer = optim.Adam(net_standard.parameters(), lr=0.0003)
        
        net_standard.train()
        for images, labels in train_batches:
            optimizer.zero_grad()
            outputs, loss = net_standard(images, labels)
            loss.backward()  # Only standard backprop
            optimizer.step()  # Only Adam
            del outputs, loss
        
        standard_acc = test_accuracy(net_standard, test_batches)
        standard_results.append(standard_acc)
        
        # Save baseline weights
        baseline_state = {name: param.data.clone() for name, param in net_standard.named_parameters()}
        del optimizer, net_standard
        clear_memory()
        
        # Method 2: Evolved only (no Adam)
        torch.manual_seed(seed)
        net_evolved = NetCustom()
        insert_parameters(net_evolved, individual)
        
        # Start from same weights
        for name, param in net_evolved.named_parameters():
            param.data.copy_(baseline_state[name])
        
        net_evolved.train()
        for images, labels in train_batches:
            net_evolved.zero_grads_g2()
            outputs, loss = net_evolved(images, labels)
            net_evolved.backprop_adv(test=False)
            
            # Only evolved updates
            adam_delta, updates = net_evolved.optimizer_step(learning_rate=0.0003)
            for p, u in zip(net_evolved.parameters(), updates):
                p.data.add_(u)  # Only evolved component
            
            del outputs, loss, adam_delta, updates
        
        evolved_only_acc = test_accuracy(net_evolved, test_batches)
        evolved_only_results.append(evolved_only_acc)
        del net_evolved
        clear_memory()
        
        # Method 3: Evolved + Adam (original implementation)
        torch.manual_seed(seed)
        net_hybrid = NetCustom()
        insert_parameters(net_hybrid, individual)
        
        # Start from same weights
        for name, param in net_hybrid.named_parameters():
            param.data.copy_(baseline_state[name])
        
        net_hybrid.train()
        for images, labels in train_batches:
            net_hybrid.zero_grads_g2()
            outputs, loss = net_hybrid(images, labels)
            net_hybrid.backprop_adv(test=False)
            
            # Both evolved and Adam updates
            adam_delta, updates = net_hybrid.optimizer_step(learning_rate=0.0003)
            for p, d, u in zip(net_hybrid.parameters(), adam_delta, updates):
                p.data.add_(u)  # Evolved component
                p.data.add_(d)  # Adam component
            
            del outputs, loss, adam_delta, updates
        
        evolved_hybrid_acc = test_accuracy(net_hybrid, test_batches)
        evolved_hybrid_results.append(evolved_hybrid_acc)
        del net_hybrid, baseline_state
        clear_memory()
    
    # Statistical analysis
    print(f"\n{dataset.upper()} Results:")
    print(f"Standard:        {np.mean(standard_results):.4f} ± {np.std(standard_results):.4f}")
    print(f"Evolved only:    {np.mean(evolved_only_results):.4f} ± {np.std(evolved_only_results):.4f}")
    print(f"Evolved + Adam:  {np.mean(evolved_hybrid_results):.4f} ± {np.std(evolved_hybrid_results):.4f}")
    
    # Statistical tests
    t_stat1, p_val1 = stats.ttest_ind(standard_results, evolved_only_results)
    t_stat2, p_val2 = stats.ttest_ind(standard_results, evolved_hybrid_results)
    t_stat3, p_val3 = stats.ttest_ind(evolved_only_results, evolved_hybrid_results)
    
    print(f"\nStatistical tests (p-values):")
    print(f"Standard vs Evolved only:   p = {p_val1:.4f} {'***' if p_val1 < 0.001 else '**' if p_val1 < 0.01 else '*' if p_val1 < 0.05 else 'ns'}")
    print(f"Standard vs Evolved+Adam:   p = {p_val2:.4f} {'***' if p_val2 < 0.001 else '**' if p_val2 < 0.01 else '*' if p_val2 < 0.05 else 'ns'}")
    print(f"Evolved only vs Evolved+Adam: p = {p_val3:.4f} {'***' if p_val3 < 0.001 else '**' if p_val3 < 0.01 else '*' if p_val3 < 0.05 else 'ns'}")
    
    return {
        'standard': standard_results,
        'evolved_only': evolved_only_results,
        'evolved_hybrid': evolved_hybrid_results,
        'p_values': [p_val1, p_val2, p_val3]
    }

def main():
    print("QUICK VALIDATION TEST")
    print("=" * 50)
    
    # Load population
    with open('population4.pkl', 'rb') as f:
        population = pickle.load(f)
    
    # Test best individual from previous results (Individual 16)
    individual_16 = population[16]
    
    print("Testing Individual 16 (previously best performer)")
    print("This will take a few minutes...")
    print()
    
    # Test on MNIST
    mnist_results = test_methods(individual_16, 'mnist')
    print()
    
    # Test on Fashion-MNIST
    fashion_results = test_methods(individual_16, 'fashion_mnist')
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    
    # Check if results hold up
    mnist_evolved_better = np.mean(mnist_results['evolved_hybrid']) > np.mean(mnist_results['standard'])
    fashion_evolved_better = np.mean(fashion_results['evolved_hybrid']) > np.mean(fashion_results['standard'])
    
    print(f"MNIST - Evolved better than standard: {mnist_evolved_better} (p={mnist_results['p_values'][1]:.4f})")
    print(f"Fashion-MNIST - Evolved better than standard: {fashion_evolved_better} (p={fashion_results['p_values'][1]:.4f})")
    
    # Critical analysis
    print("\nCRITICAL ANALYSIS:")
    if mnist_results['p_values'][1] < 0.05:
        print("✓ MNIST improvement is statistically significant")
    else:
        print("✗ MNIST improvement is NOT statistically significant")
    
    if fashion_results['p_values'][1] < 0.05:
        print("✓ Fashion-MNIST improvement is statistically significant")
    else:
        print("✗ Fashion-MNIST improvement is NOT statistically significant")
    
    # Check if evolved-only (without Adam) also works
    mnist_evolved_only_better = np.mean(mnist_results['evolved_only']) > np.mean(mnist_results['standard'])
    print(f"\nEvolved rules alone (no Adam) work on MNIST: {mnist_evolved_only_better} (p={mnist_results['p_values'][0]:.4f})")
    
    if mnist_results['p_values'][2] < 0.05:
        print("⚠️  Evolved+Adam is significantly better than Evolved-only (Adam is helping)")
    else:
        print("✓ Evolved rules work equally well with or without Adam")

if __name__ == "__main__":
    main()
