import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Path to outputs directory
output_dir = '/Users/jerrylcj/python_proj/sd-vae-med/outputs'

# Set scientific publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Models to compare
model1 = 'pt-vq-f4'
model2 = 'pt-vq-f4-noattn'
datasets = ['BUSI_2D', 'CVC_2D', 'REFUGE2-B_2D', 'SYNAPSE-B_2D', 'STS-3D_2D']

# Dictionary to store the results
all_results = {model1: {}, model2: {}}

# Collect data for both models
for model in [model1, model2]:
    model_path = os.path.join(output_dir, model)
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        continue
        
    for dataset in datasets:
        file_path = os.path.join(model_path, f"{dataset}.csv")
        if not os.path.exists(file_path):
            print(f"CSV file {file_path} does not exist.")
            continue
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        all_results[model][dataset] = df

# Prepare data for split violin plots
metrics = ['mse_2d', 'ssim_2d', 'psnr_2d']
metric_labels = ['Mean Squared Error (MSE)', 'Structural Similarity (SSIM)', 'Peak Signal-to-Noise Ratio (PSNR)']

# Create a figure with 3 rows (one for each metric)
fig, axes = plt.subplots(3, 1, figsize=(15, 18), dpi=300)
fig.suptitle('Comparison between VQ-F4 and VQ-F4-NoAttn Models', fontsize=16, fontweight='bold')

# Scientific publication color palette - colorblind friendly
# Blue for standard model, Red for noattn model
colors = ['#0173B2', '#D55E00']

# For each metric, create split violin plots for all datasets
for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[i]
    
    # For each dataset, create a split violin plot
    for j, dataset in enumerate(datasets):
        # Check if both models have data for this dataset
        if dataset in all_results[model1] and dataset in all_results[model2]:
            # Get data for both models
            data1 = all_results[model1][dataset][metric].values
            data2 = all_results[model2][dataset][metric].values
            
            # Create positions for this dataset
            pos = j
            
            # Create a dictionary for data 
            split_violin_data = {
                'Model': ['VQ-F4'] * len(data1) + ['VQ-F4-NoAttn'] * len(data2),
                'Value': np.concatenate([data1, data2]),
                'Dataset': [dataset] * (len(data1) + len(data2))
            }
            
            # Convert to DataFrame
            df_violin = pd.DataFrame(split_violin_data)
            
            # Create split violin plot for this dataset
            sns.violinplot(x='Dataset', y='Value', hue='Model', data=df_violin, 
                          ax=ax, palette=colors, split=True, inner='quartile',
                          linewidth=1, cut=0)
            
            # Add text with mean, median, and std for each model
            stats_data = {
                'VQ-F4': {'mean': np.mean(data1), 'median': np.median(data1), 'std': np.std(data1)},
                'VQ-F4-NoAttn': {'mean': np.mean(data2), 'median': np.median(data2), 'std': np.std(data2)}
            }
            
            # Format text based on metric
            if metric == 'mse_2d':
                mean_diff = stats_data['VQ-F4']['mean'] - stats_data['VQ-F4-NoAttn']['mean']
                # Add text with relative difference (%)
                rel_diff_percent = (mean_diff / stats_data['VQ-F4']['mean']) * 100
                diff_sign = "+" if rel_diff_percent < 0 else "-"  # lower MSE is better
                text = f"Δ: {diff_sign}{abs(rel_diff_percent):.1f}%"
                
                # Add at the top of the violin
                y_pos = max(np.max(data1), np.max(data2))
                ax.text(j, y_pos * 1.05, text, ha='center', va='bottom', fontsize=9,
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', alpha=0.8))
            elif metric in ['ssim_2d', 'psnr_2d']:
                mean_diff = stats_data['VQ-F4-NoAttn']['mean'] - stats_data['VQ-F4']['mean']
                # Add text with relative difference (%)
                rel_diff_percent = (mean_diff / stats_data['VQ-F4']['mean']) * 100
                diff_sign = "+" if rel_diff_percent > 0 else "-"  # higher SSIM/PSNR is better
                text = f"Δ: {diff_sign}{abs(rel_diff_percent):.1f}%"
                
                # Add at the top of the violin
                y_pos = max(np.max(data1), np.max(data2))
                ax.text(j, y_pos * 1.05, text, ha='center', va='bottom', fontsize=9,
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', alpha=0.8))
            
    # Remove the hue legend from subsequent plots (keep only one)
    if i > 0:
        ax.get_legend().remove()
    else:
        # Improve the legend for the first plot
        ax.legend(title="Model", loc='upper right', frameon=True, 
                 fancybox=True, framealpha=0.9, edgecolor='gray')
    
    # Customize the plot
    ax.set_title(metric_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_label, fontweight='bold')
    
    # Format y-axis based on metric
    if metric == 'mse_2d':
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('gray')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add a watermark or source text
fig.text(0.99, 0.01, 'VQ-F4 vs VQ-F4-NoAttn Comparison', fontsize=8, 
         color='gray', ha='right', va='bottom', alpha=0.7)

# Save the plot
output_file = os.path.join(output_dir, 'vq_f4_vs_noattn_violin.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Split violin plot saved to {output_file}")

plt.show()