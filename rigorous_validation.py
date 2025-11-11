import torch
import torch.optim as optim
import pickle
import gc
import json
import csv
import os
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from test6 import NetCustom
from load_pop3 import MNISTDataLoader

class RigorousValidator:
    def __init__(self, population_file='population4.pkl'):
        self.population_file = population_file
        self.results_dir = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Test parameters
        self.n_seeds = 10  # Multiple random seeds for statistical significance
        self.n_top_individuals = 5  # Test top N individuals
        self.datasets = ['mnist', 'fashion_mnist']
        self.batch_size = 64
        self.num_train_batches = 200  # Reasonable size for multiple runs
        self.num_test_batches = 50
        self.learning_rate = 0.0003
        self.num_epochs = 3
        
        # Load population
        with open(self.population_file, 'rb') as f:
            self.population = pickle.load(f)
            self.population = [self.population[3], self.population[7], self.population[16], self.population[17], self.population[18], self.population[28]]

        
        self.all_results = []
        
    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def insert_parameters(self, net, individual):
        """Insert evolved learning rule parameters"""
        for (layer_name, nn_dict_original), (_, nn_dict_individual) in zip(net.manager.networks.items(), individual.items()):
            for nn_name, nn_managed in nn_dict_original.items():
                individual_params = nn_dict_individual[nn_name]
                network_params = list(nn_managed.parameters())
                for param_net, param_ind in zip(network_params, individual_params):
                    param_net.data = param_ind.clone().detach()
    
    def get_batches(self, dataset_name, seed):
        """Get batches with fixed seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        fashion = (dataset_name == 'fashion_mnist')
        dataloader = MNISTDataLoader(batch_size=self.batch_size, fashion=fashion)
        
        train_batches = []
        test_batches = []
        
        for _ in range(self.num_train_batches):
            images, labels = dataloader.get_batch(mode='train')
            train_batches.append((images.clone().detach(), labels.clone().detach()))
        
        for _ in range(self.num_test_batches):
            images, labels = dataloader.get_batch(mode='eval')
            test_batches.append((images.clone().detach(), labels.clone().detach()))
        
        return train_batches, test_batches
    
    def test_accuracy(self, net, test_batches):
        """Calculate test accuracy"""
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
        
        self.clear_memory()
        return correct / total
    
    def train_standard(self, train_batches, test_batches, seed):
        """Train with standard backpropagation + Adam"""
        torch.manual_seed(seed)
        net = NetCustom()
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        
        epoch_accuracies = []
        
        for epoch in range(self.num_epochs):
            net.train()
            epoch_loss = 0
            
            for images, labels in train_batches:
                optimizer.zero_grad()
                outputs, loss = net(images, labels)
                loss.backward()  # Standard backpropagation only
                optimizer.step()  # Standard Adam only
                epoch_loss += loss.item()
                del outputs, loss
            
            accuracy = self.test_accuracy(net, test_batches)
            epoch_accuracies.append(accuracy)
            self.clear_memory()
        
        final_accuracy = epoch_accuracies[-1]
        
        # Save baseline weights for fair comparison
        baseline_state = {name: param.data.clone() for name, param in net.named_parameters()}
        del optimizer, net
        self.clear_memory()
        
        return final_accuracy, epoch_accuracies, baseline_state
    
    def train_evolved_only(self, individual, baseline_state, train_batches, test_batches, seed):
        """Train with ONLY evolved learning rules (no Adam component)"""
        torch.manual_seed(seed)
        net = NetCustom()
        self.insert_parameters(net, individual)
        
        # Start from same initial weights
        for name, param in net.named_parameters():
            param.data.copy_(baseline_state[name])
        
        epoch_accuracies = []
        
        for epoch in range(self.num_epochs):
            net.train()
            
            for images, labels in train_batches:
                net.zero_grads_g2()
                outputs, loss = net(images, labels)
                net.backprop_adv(test=False)
                
                # Apply ONLY evolved updates (no Adam delta)
                adam_delta, updates = net.optimizer_step(learning_rate=self.learning_rate)
                for p, u in zip(net.parameters(), updates):
                    p.data.add_(u)  # Only evolved updates
                
                del outputs, loss, adam_delta, updates
            
            accuracy = self.test_accuracy(net, test_batches)
            epoch_accuracies.append(accuracy)
            self.clear_memory()
        
        final_accuracy = epoch_accuracies[-1]
        del net
        self.clear_memory()
        
        return final_accuracy, epoch_accuracies
    
    def train_evolved_hybrid(self, individual, baseline_state, train_batches, test_batches, seed):
        """Train with evolved rules + Adam (original implementation)"""
        torch.manual_seed(seed)
        net = NetCustom()
        self.insert_parameters(net, individual)
        
        # Start from same initial weights
        for name, param in net.named_parameters():
            param.data.copy_(baseline_state[name])
        
        epoch_accuracies = []
        
        for epoch in range(self.num_epochs):
            net.train()
            
            for images, labels in train_batches:
                net.zero_grads_g2()
                outputs, loss = net(images, labels)
                net.backprop_adv(test=False)
                
                # Apply both evolved and Adam updates
                adam_delta, updates = net.optimizer_step(learning_rate=self.learning_rate)
                for p, d, u in zip(net.parameters(), adam_delta, updates):
                    p.data.add_(u)  # Evolved updates
                    p.data.add_(d)  # Adam updates
                
                del outputs, loss, adam_delta, updates
            
            accuracy = self.test_accuracy(net, test_batches)
            epoch_accuracies.append(accuracy)
            self.clear_memory()
        
        final_accuracy = epoch_accuracies[-1]
        del net
        self.clear_memory()
        
        return final_accuracy, epoch_accuracies
    
    def run_single_experiment(self, dataset_name, individual_idx, seed):
        """Run one complete experiment with all methods"""
        print(f"  Seed {seed}: Dataset={dataset_name}, Individual={individual_idx}")
        
        # Get data
        train_batches, test_batches = self.get_batches(dataset_name, seed)
        
        # Test standard method
        standard_acc, standard_epochs, baseline_state = self.train_standard(train_batches, test_batches, seed)
        
        # Test evolved methods
        if individual_idx >= 0:  # -1 means standard only
            individual = self.population[individual_idx]
            evolved_only_acc, evolved_only_epochs = self.train_evolved_only(individual, baseline_state, train_batches, test_batches, seed)
            evolved_hybrid_acc, evolved_hybrid_epochs = self.train_evolved_hybrid(individual, baseline_state, train_batches, test_batches, seed)
        else:
            evolved_only_acc, evolved_only_epochs = None, None
            evolved_hybrid_acc, evolved_hybrid_epochs = None, None
        
        # Store results
        result = {
            'dataset': dataset_name,
            'individual': individual_idx,
            'seed': seed,
            'standard_acc': standard_acc,
            'evolved_only_acc': evolved_only_acc,
            'evolved_hybrid_acc': evolved_hybrid_acc,
            'standard_epochs': standard_epochs,
            'evolved_only_epochs': evolved_only_epochs,
            'evolved_hybrid_epochs': evolved_hybrid_epochs
        }
        
        self.all_results.append(result)
        
        # Clean up
        del train_batches, test_batches, baseline_state
        self.clear_memory()
        
        return result
    
    def run_statistical_validation(self):
        """Run comprehensive statistical validation"""
        print("=" * 80)
        print("RIGOROUS VALIDATION FOR PUBLICATION")
        print("=" * 80)
        print(f"Testing {self.n_seeds} seeds × {len(self.datasets)} datasets × {self.n_top_individuals} individuals")
        print(f"Results will be saved to: {self.results_dir}/")
        print()
        
        # First, run quick screening to find top individuals on MNIST
        print("Phase 1: Screening top individuals on MNIST...")
        screening_results = []
        
        for i in range(len(self.population)):
            result = self.run_single_experiment('mnist', i, seed=42)  # Fixed seed for screening
            screening_results.append((i, result['evolved_hybrid_acc']))
            if i % 5 == 0:
                print(f"  Screened {i+1}/{len(self.population)} individuals")
        
        # Sort by performance and select top N
        screening_results.sort(key=lambda x: x[1], reverse=True)
        top_individuals = [idx for idx, _ in screening_results[:self.n_top_individuals]]
        
        print(f"Top {self.n_top_individuals} individuals: {top_individuals}")
        print(f"Their screening accuracies: {[acc for _, acc in screening_results[:self.n_top_individuals]]}")
        print()
        
        # Phase 2: Rigorous testing
        print("Phase 2: Rigorous multi-seed testing...")
        
        total_experiments = len(self.datasets) * (self.n_top_individuals + 1) * self.n_seeds  # +1 for standard baseline
        experiment_count = 0
        
        for dataset_name in self.datasets:
            print(f"\nTesting on {dataset_name.upper()}:")
            
            # Test standard baseline
            for seed in range(self.n_seeds):
                experiment_count += 1
                print(f"  Progress: {experiment_count}/{total_experiments}")
                self.run_single_experiment(dataset_name, -1, seed)  # -1 = standard only
            
            # Test top individuals
            for individual_idx in top_individuals:
                for seed in range(self.n_seeds):
                    experiment_count += 1
                    print(f"  Progress: {experiment_count}/{total_experiments}")
                    self.run_single_experiment(dataset_name, individual_idx, seed)
        
        print("\nPhase 3: Statistical analysis...")
        self.analyze_results()
        
        print(f"\nAll results saved to: {self.results_dir}/")
    
    def analyze_results(self):
        """Perform statistical analysis"""
        # Save raw results
        with open(f"{self.results_dir}/raw_results.json", 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Create CSV for easy analysis
        csv_path = f"{self.results_dir}/results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.all_results[0].keys())
            writer.writeheader()
            writer.writerows(self.all_results)
        
        print(f"Raw results saved to {csv_path}")
        
        # Statistical analysis
        analysis_results = {}
        
        for dataset in self.datasets:
            analysis_results[dataset] = {}
            
            # Get standard baseline results
            standard_results = [r['standard_acc'] for r in self.all_results 
                              if r['dataset'] == dataset and r['individual'] == -1]
            
            print(f"\n{dataset.upper()} Results:")
            print(f"Standard method: {np.mean(standard_results):.4f} ± {np.std(standard_results):.4f}")
            
            analysis_results[dataset]['standard'] = {
                'mean': np.mean(standard_results),
                'std': np.std(standard_results),
                'values': standard_results
            }
            
            # Analyze each individual
            for individual_idx in range(len(self.population)):
                individual_results = [r for r in self.all_results 
                                    if r['dataset'] == dataset and r['individual'] == individual_idx]
                
                if not individual_results:
                    continue
                
                # Evolved only results
                evolved_only = [r['evolved_only_acc'] for r in individual_results if r['evolved_only_acc'] is not None]
                evolved_hybrid = [r['evolved_hybrid_acc'] for r in individual_results if r['evolved_hybrid_acc'] is not None]
                
                if evolved_only and evolved_hybrid:
                    # Statistical tests
                    t_stat_only, p_val_only = stats.ttest_ind(standard_results, evolved_only)
                    t_stat_hybrid, p_val_hybrid = stats.ttest_ind(standard_results, evolved_hybrid)
                    
                    print(f"Individual {individual_idx}:")
                    print(f"  Evolved only:   {np.mean(evolved_only):.4f} ± {np.std(evolved_only):.4f} (p={p_val_only:.4f})")
                    print(f"  Evolved+Adam:   {np.mean(evolved_hybrid):.4f} ± {np.std(evolved_hybrid):.4f} (p={p_val_hybrid:.4f})")
                    
                    analysis_results[dataset][f'individual_{individual_idx}'] = {
                        'evolved_only_mean': np.mean(evolved_only),
                        'evolved_only_std': np.std(evolved_only),
                        'evolved_hybrid_mean': np.mean(evolved_hybrid),
                        'evolved_hybrid_std': np.std(evolved_hybrid),
                        'p_value_only': p_val_only,
                        'p_value_hybrid': p_val_hybrid,
                        'significant_only': p_val_only < 0.05,
                        'significant_hybrid': p_val_hybrid < 0.05
                    }
        
        # Save analysis
        with open(f"{self.results_dir}/statistical_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Summary report
        with open(f"{self.results_dir}/summary_report.txt", 'w') as f:
            f.write("STATISTICAL VALIDATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset in self.datasets:
                f.write(f"{dataset.upper()} RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                standard_mean = analysis_results[dataset]['standard']['mean']
                f.write(f"Standard baseline: {standard_mean:.4f} ± {analysis_results[dataset]['standard']['std']:.4f}\n\n")
                
                significant_improvements = []
                for key, data in analysis_results[dataset].items():
                    if key.startswith('individual_'):
                        individual_idx = key.split('_')[1]
                        if data.get('significant_hybrid', False):
                            improvement = data['evolved_hybrid_mean'] - standard_mean
                            significant_improvements.append((individual_idx, improvement, data['p_value_hybrid']))
                
                if significant_improvements:
                    f.write("STATISTICALLY SIGNIFICANT IMPROVEMENTS:\n")
                    for idx, improvement, p_val in sorted(significant_improvements, key=lambda x: x[1], reverse=True):
                        f.write(f"Individual {idx}: +{improvement:.4f} (p={p_val:.4f})\n")
                else:
                    f.write("NO STATISTICALLY SIGNIFICANT IMPROVEMENTS FOUND\n")
                
                f.write("\n")
        
        print(f"Statistical analysis saved to {self.results_dir}/statistical_analysis.json")
        print(f"Summary report saved to {self.results_dir}/summary_report.txt")

if __name__ == "__main__":
    validator = RigorousValidator()
    validator.run_statistical_validation()
