import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(dataset_path):
    # Load the data from the CSV file
    df = pd.read_csv(dataset_path, header=None)

    # Count the number of columns
    num_features = df.shape[1]

    # Create an adequate number of features
    features = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(num_features - 2)]

    # Assign the features as column names
    df.columns = features

    # Create a subplot with a grid of histograms
    num_cols = 6
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        if i == 0:
            continue  # Skip the ID column
        sns.histplot(data=df, x=feature, hue='Diagnosis', ax=axes[i - 1], kde=True)
        axes[i].set_title(feature)

    # Hide any unused subplots
    for j in range(i, len(axes)):
        fig.delaxes(axes[j])

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

def main():
    parser = argparse.ArgumentParser(description="Visualize a dataset with histograms and pair plots.")
    parser.add_argument("-d", "--dataset", required=True, help="Path to the dataset (CSV file) to be visualized.")
    args = parser.parse_args()

    visualize_data(args.dataset)

if __name__ == "__main__":
    main()