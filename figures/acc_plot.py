
import matplotlib.pyplot as plt
import pandas as pd
import os


datasets = [name for name in os.listdir('../training_data') if os.path.isdir(os.path.join('../training_data', name))]


def plot_acc(dataset):
    # Load dataset
    acc_df = pd.read_csv(f"../acc_rate_csvs/{dataset}.csv")
    dataset_size = pd.read_csv(f"../training_data/{dataset}/folds.csv").shape[0]

    # Define a fixed order for the methods
    fixed_method_order = sorted(acc_df['method'].unique())

    # Create a figure with two subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 2))  # (1 row, 2 columns)

    ### First plot: Accuracy per fold ###
    for method in fixed_method_order:
        if method in acc_df['method'].unique():
            method_data = acc_df[acc_df['method'] == method]
            
            # Scatter plot (all methods treated the same)
            ax[0].scatter(method_data['acc'], [method]*len(method_data), s=50, marker='D', color='white', edgecolor='black', linewidth=1)

    # Add labels for the first plot
    ax[0].set_xlabel('Accuracy (acc)')
    ax[0].set_ylabel('Method')
    ax[0].set_yticks(fixed_method_order)  # Set y-ticks to fixed method order
    ax[0].grid(True)

    ### Second plot: Mean accuracy and standard deviation ###
    # Compute mean and standard deviation for each method
    method_stats = acc_df.groupby('method')['acc'].agg(['mean', 'std']).reset_index()
    method_stats.columns = ['method', 'mean', 'std']  # Rename columns

    # Filter method_stats to include only the fixed order
    method_stats = method_stats[method_stats['method'].isin(fixed_method_order)]
    method_stats = method_stats.set_index('method').reindex(fixed_method_order).reset_index()

    # Plot the mean as a point and the standard deviation as error bars (all methods treated the same)
    ax[1].errorbar(method_stats['mean'], method_stats['method'], xerr=0.25 * method_stats['std'], fmt='D', color='black', ecolor='black', elinewidth=1, capsize=4, markersize=6)

    # Add labels for the second plot
    ax[1].set_xlabel('Accuracy (acc)')
    ax[1].set_ylabel('Method')
    ax[1].set_yticks(fixed_method_order)  # Set y-ticks to fixed method order
    ax[1].grid(True)

    # Add a single overall title for the entire figure
    fig.suptitle(f'Accuracy Analysis for {dataset} (Dataset Size: {dataset_size})')

    # Save the figure as a PNG file
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'pngs/{dataset}.png', bbox_inches='tight')  # Save the figure as PNG
    plt.close(fig)  # Close the figure to free memory


for dataset in datasets:
    plot_acc(dataset)