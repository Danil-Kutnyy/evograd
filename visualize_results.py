#!/usr/bin/env python3
"""
Comprehensive visualization script for validation results.
This script creates multiple plots to analyze the experimental results from raw_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(filepath):
    """Load and parse the JSON results file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def filter_data(data, dataset=None, exclude_control=True):
    """Filter data by dataset and optionally exclude control group (individual -1)"""
    filtered = data.copy()
    
    if dataset:
        filtered = [d for d in filtered if d['dataset'] == dataset]
    
    if exclude_control:
        filtered = [d for d in filtered if d['individual'] != -1]
    
    return filtered

def create_dataframe(data):
    """Convert raw data to pandas DataFrame for easier analysis"""
    records = []
    
    for entry in data:
        # Extract base info
        base_info = {
            'dataset': entry['dataset'],
            'individual': entry['individual'],
            'seed': entry['seed']
        }
        
        # Add standard method results
        if entry['standard_acc'] is not None:
            record = base_info.copy()
            record.update({
                'method': 'Standard',
                'final_accuracy': entry['standard_acc'],
                'epoch_accuracies': entry['standard_epochs']
            })
            records.append(record)
        
        # Add evolved-only method results
        if entry['evolved_only_acc'] is not None:
            record = base_info.copy()
            record.update({
                'method': 'Evolved Only',
                'final_accuracy': entry['evolved_only_acc'],
                'epoch_accuracies': entry['evolved_only_epochs']
            })
            records.append(record)
        
        # Add evolved-hybrid method results
        if entry['evolved_hybrid_acc'] is not None:
            record = base_info.copy()
            record.update({
                'method': 'Evolved Hybrid',
                'final_accuracy': entry['evolved_hybrid_acc'],
                'epoch_accuracies': entry['evolved_hybrid_epochs']
            })
            records.append(record)
    
    return pd.DataFrame(records)

def plot_method_comparison(df, save_path=None):
    """Create box plots comparing methods across datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    datasets = df['dataset'].unique()
    
    for i, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]
        
        # Only include non-control individuals for comparison
        dataset_data = dataset_data[dataset_data['individual'] != -1]
        
        sns.boxplot(data=dataset_data, x='method', y='final_accuracy', ax=axes[i])
        axes[i].set_title(f'{dataset.upper()} - Method Comparison')
        axes[i].set_ylabel('Final Accuracy')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add mean values as annotations
        for j, method in enumerate(dataset_data['method'].unique()):
            method_data = dataset_data[dataset_data['method'] == method]
            mean_acc = method_data['final_accuracy'].mean()
            axes[i].text(j, mean_acc + 0.005, f'{mean_acc:.4f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_summary(df, save_path=None):
    """Create bar plots with error bars showing mean and std"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    datasets = df['dataset'].unique()
    
    for i, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]
        dataset_data = dataset_data[dataset_data['individual'] != -1]
        
        # Calculate statistics
        stats_summary = dataset_data.groupby('method')['final_accuracy'].agg(['mean', 'std']).reset_index()
        
        bars = axes[i].bar(stats_summary['method'], stats_summary['mean'], 
                          yerr=stats_summary['std'], capsize=5, alpha=0.7)
        
        axes[i].set_title(f'{dataset.upper()} - Mean Performance ± Std')
        axes[i].set_ylabel('Accuracy')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, stats_summary['mean'], stats_summary['std']):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + std_val + 0.002,
                        f'{mean_val:.4f}±{std_val:.4f}',
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves(data, dataset='mnist', save_path=None):
    """Plot training curves showing accuracy over epochs"""
    dataset_data = filter_data(data, dataset=dataset, exclude_control=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Get unique individuals (including control)
    individuals = sorted(list(set(d['individual'] for d in dataset_data)))
    
    for idx, individual in enumerate(individuals[:6]):  # Plot first 6 individuals
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        individual_data = [d for d in dataset_data if d['individual'] == individual]
        
        # Group by method and plot averaged curves
        methods = {}
        if individual != -1:  # Non-control individuals have all methods
            for entry in individual_data:
                if entry['standard_epochs']:
                    if 'Standard' not in methods:
                        methods['Standard'] = []
                    methods['Standard'].append(entry['standard_epochs'])
                
                if entry['evolved_only_epochs']:
                    if 'Evolved Only' not in methods:
                        methods['Evolved Only'] = []
                    methods['Evolved Only'].append(entry['evolved_only_epochs'])
                
                if entry['evolved_hybrid_epochs']:
                    if 'Evolved Hybrid' not in methods:
                        methods['Evolved Hybrid'] = []
                    methods['Evolved Hybrid'].append(entry['evolved_hybrid_epochs'])
        else:  # Control individual only has standard method
            for entry in individual_data:
                if entry['standard_epochs']:
                    if 'Standard (Control)' not in methods:
                        methods['Standard (Control)'] = []
                    methods['Standard (Control)'].append(entry['standard_epochs'])
        
        # Plot curves for each method
        epochs = range(1, 4)  # 3 epochs based on the data
        for method, curves in methods.items():
            if curves:
                curves_array = np.array(curves)
                mean_curve = np.mean(curves_array, axis=0)
                std_curve = np.std(curves_array, axis=0)
                
                ax.plot(epochs, mean_curve, label=method, marker='o')
                ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        
        title = f'Individual {individual}' if individual != -1 else 'Control Group'
        ax.set_title(f'{dataset.upper()} - {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(individuals), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Training Curves - {dataset.upper()}', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_heatmap_performance(df, dataset='mnist', save_path=None):
    """Create heatmap showing performance across individuals and seeds"""
    dataset_data = df[df['dataset'] == dataset]
    dataset_data = dataset_data[dataset_data['individual'] != -1]  # Exclude control
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = ['Standard', 'Evolved Only', 'Evolved Hybrid']
    
    for i, method in enumerate(methods):
        method_data = dataset_data[dataset_data['method'] == method]
        
        # Create pivot table for heatmap
        pivot_data = method_data.pivot_table(
            values='final_accuracy', 
            index='individual', 
            columns='seed', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis', 
                   ax=axes[i], cbar_kws={'label': 'Accuracy'})
        axes[i].set_title(f'{method}')
        axes[i].set_xlabel('Seed')
        axes[i].set_ylabel('Individual')
    
    plt.suptitle(f'Performance Heatmap - {dataset.upper()}', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def statistical_analysis(df):
    """Perform statistical tests and print summary"""
    print("=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)
    
    for dataset in df['dataset'].unique():
        print(f"\n{dataset.upper()} Dataset Analysis:")
        print("-" * 40)
        
        dataset_data = df[df['dataset'] == dataset]
        dataset_data = dataset_data[dataset_data['individual'] != -1]  # Exclude control
        
        # Summary statistics
        summary = dataset_data.groupby('method')['final_accuracy'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(f"\nSummary Statistics:")
        print(summary.round(4))
        
        # Pairwise comparisons using t-tests
        methods = dataset_data['method'].unique()
        print(f"\nPairwise t-test comparisons (p-values):")
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = dataset_data[dataset_data['method'] == method1]['final_accuracy']
                data2 = dataset_data[dataset_data['method'] == method2]['final_accuracy']
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"{method1} vs {method2}: p = {p_value:.6f} {significance}")
        
        # ANOVA test
        method_groups = [dataset_data[dataset_data['method'] == method]['final_accuracy'] 
                        for method in methods]
        f_stat, p_anova = stats.f_oneway(*method_groups)
        print(f"\nANOVA F-test: F = {f_stat:.4f}, p = {p_anova:.6f}")
        
        # Effect sizes (Cohen's d)
        print(f"\nEffect sizes (Cohen's d):")
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = dataset_data[dataset_data['method'] == method1]['final_accuracy']
                data2 = dataset_data[dataset_data['method'] == method2]['final_accuracy']
                
                pooled_std = np.sqrt(((len(data1) - 1) * data1.std()**2 + 
                                    (len(data2) - 1) * data2.std()**2) / 
                                   (len(data1) + len(data2) - 2))
                cohens_d = (data1.mean() - data2.mean()) / pooled_std
                
                print(f"{method1} vs {method2}: d = {cohens_d:.4f}")

def save_summary_table(df, save_path=None):
    """Save a summary table to CSV"""
    summary_stats = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        dataset_data = dataset_data[dataset_data['individual'] != -1]
        
        for method in dataset_data['method'].unique():
            method_data = dataset_data[dataset_data['method'] == method]['final_accuracy']
            
            summary_stats.append({
                'Dataset': dataset,
                'Method': method,
                'Count': len(method_data),
                'Mean': method_data.mean(),
                'Std': method_data.std(),
                'Min': method_data.min(),
                'Max': method_data.max(),
                'Median': method_data.median(),
                'Q1': method_data.quantile(0.25),
                'Q3': method_data.quantile(0.75)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"Summary table saved to: {save_path}")
    
    return summary_df

def main():
    """Main function to run all visualizations"""
    # Load data
    data_path = "validation_results_20250819_131745/raw_results.json"
    
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found!")
        return
    
    print("Loading data...")
    data = load_data(data_path)
    
    print("Converting to DataFrame...")
    df = create_dataframe(data)
    
    print(f"Data shape: {df.shape}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Individuals: {sorted(df['individual'].unique())}")
    
    # Create output directory
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # 1. Method comparison box plots
    print("1. Method comparison box plots...")
    plot_method_comparison(df, save_path=output_dir / "method_comparison.png")
    
    # 2. Statistical summary with error bars
    print("2. Statistical summary...")
    plot_statistical_summary(df, save_path=output_dir / "statistical_summary.png")
    
    # 3. Training curves for each dataset
    print("3. Training curves...")
    for dataset in df['dataset'].unique():
        plot_training_curves(data, dataset=dataset, 
                           save_path=output_dir / f"training_curves_{dataset}.png")
    
    # 4. Performance heatmaps
    print("4. Performance heatmaps...")
    for dataset in df['dataset'].unique():
        plot_heatmap_performance(df, dataset=dataset,
                                save_path=output_dir / f"heatmap_{dataset}.png")
    
    # 5. Statistical analysis
    print("5. Statistical analysis...")
    statistical_analysis(df)
    
    # 6. Save summary table
    print("6. Saving summary table...")
    summary_df = save_summary_table(df, save_path=output_dir / "summary_statistics.csv")
    print("\nSummary Statistics:")
    print(summary_df.round(4))
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()


