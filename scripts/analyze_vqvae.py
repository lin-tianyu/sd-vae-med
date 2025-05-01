import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Path to outputs directory
output_dir = '/Users/jerrylcj/python_proj/sd-vae-med/outputs'

# Set scientific publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Collect VQ models data - only include standard models, not the variants
vq_models = ['pt-vq-f4', 'pt-vq-f8', 'pt-vq-f16']  # Removed pt-vq-f4-noattn and pt-vq-f8-n256
datasets = ['BUSI_2D', 'CVC_2D', 'OCT2017_2D', 'MIMIC_2D', 'REFUGE2-B_2D', 'SYNAPSE-B_2D', 'STS-3D_2D']

# Extract the downsampling factor numbers for sorting
factors = [int(re.search(r'f(\d+)', model).group(1)) for model in vq_models]
# Create a mapping of model to factor for later use
model_to_factor = {model: factor for model, factor in zip(vq_models, factors)}

# Dictionary to store the results
results_vq = {}

# Loop through models and datasets
for model in vq_models:
    model_path = os.path.join(output_dir, model)
    if not os.path.exists(model_path):
        continue
        
    results_vq[model] = {}
    
    for dataset in datasets:
        file_path = os.path.join(model_path, f"{dataset}.csv")
        if not os.path.exists(file_path):
            continue
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Calculate the average metrics and standard deviations
        results_vq[model][dataset] = {
            'mse': {
                'mean': df['mse_2d'].mean(), 
                'std': df['mse_2d'].std()
            },
            'ssim': {
                'mean': df['ssim_2d'].mean(), 
                'std': df['ssim_2d'].std()
            },
            'psnr': {
                'mean': df['psnr_2d'].mean(), 
                'std': df['psnr_2d'].std()
            }
        }

# Prepare data for plotting
models_list = sorted(list(results_vq.keys()), key=lambda x: model_to_factor[x])
datasets_list = datasets.copy()

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
fig.suptitle('VQ-VAE Models Performance by Downsampling Factor', fontsize=16, fontweight='bold')

# Set up metrics and labels
metrics = ['mse', 'ssim', 'psnr']
metric_labels = ['Mean Squared Error (MSE)', 'Structural Similarity (SSIM)', 'Peak Signal-to-Noise Ratio (PSNR)']

# Scientific publication color palette - colorblind friendly
# Using a color palette inspired by Nature, Science, etc.
colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#661D98', '#E76BF3']
markers = ['o', 's', '^', 'D', 'p', '*', 'X']
linestyles = ['-', '--', ':', '-.', '-', '--', ':']

# For each metric, create a line plot
for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[i]
    
    # For each dataset, plot a line across all models
    for j, dataset in enumerate(datasets_list):
        # Extract x values (downsampling factors) and y values (metric values)
        x_values = []
        y_values = []
        yerr_values = []  # Standard deviation values for error bars
        
        for model in models_list:
            if dataset in results_vq[model]:
                x_values.append(model_to_factor[model])
                y_values.append(results_vq[model][dataset][metric]['mean'])
                yerr_values.append(results_vq[model][dataset][metric]['std'])
        
        # Only plot if we have data points
        if len(x_values) > 0:
            # Plot line with markers and error bars
            ax.errorbar(x_values, y_values, yerr=yerr_values, label=dataset,
                      fmt=markers[j], linestyle=linestyles[j], 
                      color=colors[j], linewidth=2, markersize=8,
                      capsize=5, capthick=1, elinewidth=1)
    
    # Customize the plot
    ax.set_title(metric_label, fontsize=14, fontweight='bold')
    ax.set_xlabel('Downsampling Factor', fontweight='bold')
    ax.set_ylabel(metric_label, fontweight='bold')
    
    # Set x-axis to show only the actual downsampling factors
    actual_factors = sorted(list(set([model_to_factor[model] for model in models_list])))
    ax.set_xticks(actual_factors)
    ax.set_xticklabels([str(f) for f in actual_factors])
    
    # Dynamically adjust y-axis limits for better visualization
    if metric == 'mse':
        # Use scientific notation for MSE
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        # For MSE, lower is better, but don't invert axis
        y_values_all = []
        yerr_values_all = []
        for dataset in datasets_list:
            for model in models_list:
                if dataset in results_vq[model]:
                    y_values_all.append(results_vq[model][dataset][metric]['mean'])
                    yerr_values_all.append(results_vq[model][dataset][metric]['std'])
        
        if y_values_all:
            # Set y limits with some padding to accommodate error bars
            min_val = min([y - e for y, e in zip(y_values_all, yerr_values_all)])
            max_val = max([y + e for y, e in zip(y_values_all, yerr_values_all)])
            padding = (max_val - min_val) * 0.1
            ax.set_ylim(max(0, min_val - padding), max_val + padding)
            
    elif metric == 'ssim':
        # For SSIM, higher is better (range 0-1)
        y_values_all = []
        yerr_values_all = []
        for dataset in datasets_list:
            for model in models_list:
                if dataset in results_vq[model]:
                    y_values_all.append(results_vq[model][dataset][metric]['mean'])
                    yerr_values_all.append(results_vq[model][dataset][metric]['std'])
        
        if y_values_all:
            # Set y limits with some padding, but keep within 0-1 range
            min_val = min([y - e for y, e in zip(y_values_all, yerr_values_all)])
            max_val = max([y + e for y, e in zip(y_values_all, yerr_values_all)])
            padding = (max_val - min_val) * 0.1
            ax.set_ylim(max(0, min_val - padding), min(1, max_val + padding))
    
    elif metric == 'psnr':
        # For PSNR, higher is better
        y_values_all = []
        yerr_values_all = []
        for dataset in datasets_list:
            for model in models_list:
                if dataset in results_vq[model]:
                    y_values_all.append(results_vq[model][dataset][metric]['mean'])
                    yerr_values_all.append(results_vq[model][dataset][metric]['std'])
        
        if y_values_all:
            # Set y limits with some padding
            min_val = min([y - e for y, e in zip(y_values_all, yerr_values_all)])
            max_val = max([y + e for y, e in zip(y_values_all, yerr_values_all)])
            padding = (max_val - min_val) * 0.1
            ax.set_ylim(max(0, min_val - padding), max_val + padding)
        
    # Grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend in each subplot for better visibility
    legend = ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, framealpha=0.9, 
                       edgecolor='gray', facecolor='white')
    legend.get_frame().set_linewidth(0.5)
    
    # Add box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('gray')

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Add a watermark or source text
fig.text(0.99, 0.01, 'VQ-VAE Model Comparison', fontsize=8, 
         color='gray', ha='right', va='bottom', alpha=0.7)

# Save the plot
output_file = os.path.join(output_dir, 'vqvae_comparison.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"VQ-VAE plot saved to {output_file}")

plt.show()