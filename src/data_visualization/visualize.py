from pyexpat import features

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the data from the CSV file
df = pd.read_csv('../../data.csv', header=None)

# Count the number of columns
num_features = df.shape[1]

# Create an adequate number of features
features[0] = 'ID'
features[1] = 'Diagnosis'
features[2:] = [f'feature_{i}' for i in range(num_features - 2)]

# Assign the features as column names
df.columns = features

# Create a subplot with a grid of histograms
num_cols = 6
num_rows = (num_features - 2 + num_cols - 1) // num_cols  # Calculate the number of rows needed
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(data=df, x=feature, hue='Diagnosis', ax=axes[i], kde=True)
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()


# Create a pair plot
g = sns.pairplot(df, hue='Diagnosis', vars=[f'feature_{i}' for i in range(num_features - 2)], plot_kws={'s': 3}, dropna=True)

# Adjust the subplot parameters to give more space for the legend and column titles
g.fig.subplots_adjust(bottom=0.05, left=0.1, right=0.93, top=0.95, hspace=0.2)

# Adjust the y-axis labels
for ax in g.axes[:, 0]:
    ax.set_ylabel(ax.get_ylabel(), rotation=0, ha='right', fontsize=5)

plt.show()