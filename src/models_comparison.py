"""
This script plots and compares the training history of different trained models.

It reads training and validation history files and generates plots for various metrics.

Dependencies:
    - numpy
    - glob
    - sys
    - os
    - argparse
    - matplotlib.pyplot
"""

import numpy as np
import glob
import sys
import os
import argparse
import matplotlib.pyplot as plt

def main():
    """
    Main function to visualize and compare the training history of multiple models.
    
    The script takes an optional argument:
        - model_names: Names of the models to plot (leave empty to plot all).
    """
    # Create an argument parser with a proper description
    parser = argparse.ArgumentParser(description='Plot training history of models')

    # Add optional argument for specifying model names
    parser.add_argument('model_names', nargs='*', help='Names of the models to plot (leave empty to plot all)')

    # Parse arguments
    args = parser.parse_args()

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(script_dir, '../models')

    # Get filenames
    # All model if no argument is passed
    if len(sys.argv) == 1:
        filenames_training_history_array = glob.glob(f'{models_path}/*/*_training_history.npy')
        filenames_validate_history_array = glob.glob(f'{models_path}/*/*_validate_history.npy')
        # Create dictionaries with model name as key (filename.split('/')[-2] is the model name by folder normalization)
        filenames_training_history = {filename.split('/')[-2]: filename for filename in filenames_training_history_array}
        filenames_validate_history = {filename.split('/')[-2]: filename for filename in filenames_validate_history_array}
    else:
        filenames_training_history = {}
        filenames_validate_history = {}
        for model_name in sys.argv[1:]:
            if not os.path.exists(f'{script_dir}/../models/{model_name}'):
                print(f"Warning: Model not found: {model_name}")
                continue
            filenames_training_history[model_name] = f'{models_path}/{model_name}/{model_name}_training_history.npy'
            filenames_validate_history[model_name] = f'{models_path}/{model_name}/{model_name}_validate_history.npy'

    num_metrics = 0
    sample_history = None
    # find the history with the most metrics
    for model_name in filenames_training_history.keys():
        sample_history = np.load(filenames_training_history[model_name], allow_pickle=True).item()
        if len(sample_history.keys()) > num_metrics:
            num_metrics = len(sample_history.keys())

    nb_cols = min(4, num_metrics + 1)  # Max 4 columns per row
    nb_rows = (num_metrics + 1 + nb_cols - 1) // nb_cols  # Compute number of rows needed

    # Create a figure and subplots
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(nb_cols * 4, nb_rows * 3))
    axes = axes.flatten()  # Flatten to index easily

    # Loop through models
    for model_name in filenames_training_history.keys():
        try:
            training_history = np.load(filenames_training_history[model_name], allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Unknown model name: {model_name}")
            continue

        validate_history = np.load(filenames_validate_history[model_name], allow_pickle=True).item()

        if sample_history is None:
            sample_history = training_history

        # Loss subplot (first one)
        line_train, = axes[0].plot(training_history['loss'][:], label=model_name + ' training')  # Store line object
        color = line_train.get_color()  # Get color assigned by Matplotlib
        axes[0].plot(validate_history['loss'][:], linestyle='--', color=color, label=model_name + ' validate')  # Use same color
        axes[0].set_title('Loss')
        axes[0].legend()

        # Other metric subplots
        for j, metric in enumerate(sample_history.keys()):
            if metric == 'loss':
                continue
            if training_history.get(metric) is not None:
                line_train, = axes[j + 1].plot(training_history.get(metric)[:], label=model_name + ' training')
                color = line_train.get_color()  # Get the color assigned
                axes[j + 1].plot(validate_history.get(metric)[:], linestyle='--', color=color, label=model_name + ' validate')
                axes[j + 1].set_title(f'{metric.title()}')
                axes[j + 1].legend()

    # Hide unused subplots
    for i in range(len(sample_history.keys()), nb_rows * nb_cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An exception occurred: {e}")
        sys.exit(1)
