# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
loss_type = 'square'

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a single figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

# Plot the data for each model
for i, model in enumerate(['rnn', 'lstm', 'gru']):
    stat_df = pd.read_csv(f"../model/{model}/stat.csv")
    stat_df = stat_df[stat_df['loss_type'] == loss_type]
    stat_compress_df = stat_df.groupby(['compress_type', 'compress_size'])['val_loss'].mean().reset_index()
    stat_compress_df['compress_size'] = stat_compress_df['compress_size'].astype(str)
    
    # Apply log transformation to val_loss
    stat_compress_df['log_val_loss'] = np.log10(stat_compress_df['val_loss'])

    # Plot on the i-th axis (subplot), keep the legend enabled for the first plot only
    sns.lineplot(data=stat_compress_df, x='compress_size', y='log_val_loss', hue='compress_type', 
                 marker='o', ax=axes[i])

    # Disable the legend for each individual plot
    axes[i].get_legend().set_visible(False)

    # Add labels and title for each subplot
    axes[i].set_xlabel("Padding Window Size")
    axes[i].set_ylabel("Log of Average Val Loss")
    axes[i].set_title(f"{model}", loc='left')
    axes[i].grid(True)

# Instead of creating handles manually, we use the first plot to generate the legend
handles, labels = axes[0].get_legend_handles_labels()

# Add the global legend at the top center of the figure
fig.legend(handles, labels, title='Padding method', loc='upper right', ncol=3)

# Adjust layout and ensure the legend doesn't overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the space needed for the legend

# Display the plot
plt.savefig('analyze_pngs/padding_window_size_vs_log_val_loss.png')


