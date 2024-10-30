import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs('pngs', exist_ok=True)

datasets = [name for name in os.listdir('../training_data') if os.path.isdir(os.path.join('../training_data', name))]


def plot_acc(dataset):
    # Load dataset
    acc_df = pd.read_csv(f"../acc_rate_csvs/{dataset}.csv")
    dataset_size = pd.read_csv(f"../training_data/{dataset}/folds.csv").shape[0]
    dataset_length_min = pd.read_csv(f"../training_data/{dataset}/features.csv")['count'].min()
    dataset_length_max = pd.read_csv(f"../training_data/{dataset}/features.csv")['count'].max()


    # Define a fixed order for the methods
    fixed_method_order = sorted(acc_df['method'].unique())


    # Create a figure with two subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, max(4, acc_df['method'].nunique() / 5)))  # Ensure minimum height


    ### First plot: Accuracy per fold ###
    for method in fixed_method_order:
        if method in acc_df['method'].unique():
            method_data = acc_df[acc_df['method'] == method]
           
            # Set color to red if method contains 'gru' or 'lstm', otherwise black
            color = 'red' if "gru" in method or "lstm" in method or "rnn" in method or "mlp" in method else 'black'
           
            # Scatter plot (all methods treated the same)
            ax[0].scatter(method_data['acc'], [method]*len(method_data), s=50, marker='D', color='white', edgecolor=color, linewidth=1)


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


    # Plot the mean as a point and the standard deviation as error bars
    for index, row in method_stats.iterrows():
        color = 'red' if "gru" in row['method'] or "lstm" in row['method'] or "rnn" in row['method'] or "mlp" in row['method'] else 'black'
        ax[1].errorbar(row['mean'], row['method'], xerr=0.25 * row['std'], fmt='D', color=color, ecolor=color, elinewidth=1, capsize=4, markersize=6)


    # Add labels for the second plot
    ax[1].set_xlabel('Accuracy (acc)')
    ax[1].set_ylabel('Method')
    ax[1].set_yticks(fixed_method_order)  # Set y-ticks to fixed method order
    ax[1].grid(True)


    # Add a single overall title for the entire figure
    fig.suptitle(f'{dataset} (Dataset Size: {dataset_size} -- length {dataset_length_min} - {dataset_length_max})')


    # Save the figure as a PNG file
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'pngs/{dataset}.png', bbox_inches='tight')  # Save the figure as PNG
    plt.close(fig)  # Close the figure to free memory


# Run the function for each dataset
for dataset in datasets:
    plot_acc(dataset)