from random import sample

import numpy as np
import glob
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Get filenames
    # All model if no argument is passed
    if len(sys.argv) == 1:
        filenames_training_history = glob.glob('*_training_history.npy')
        filenames_validate_history = glob.glob('*_validate_history.npy')
    else:
        filenames_training_history = []
        filenames_validate_history = []
        for model_name in sys.argv[1:]:
            filenames_training_history.append(model_name + '_training_history.npy')
            filenames_validate_history.append(model_name + '_validate_history.npy')

    num_metrics = 0
    sample_history = np.load(filenames_training_history[0], allow_pickle=True).item()
    # find the history with the most metrics
    for filename in filenames_training_history:
        sample_history = np.load(filename, allow_pickle=True).item()
        if len(sample_history.keys()) > num_metrics:
            num_metrics = len(sample_history.keys())

    nb_cols = min(4, num_metrics + 1)  # Max 4 columns per row
    nb_rows = (num_metrics + 1 + nb_cols - 1) // nb_cols  # Compute number of rows needed

    # Create a figure and subplots
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(nb_cols * 4, nb_rows * 3))
    axes = axes.flatten()  # Flatten to index easily

    # Loop through models
    for filename in filenames_training_history:
        model_name = filename.split('_')[0]
        try:
            training_history = np.load(filename, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Unknown model name: {model_name}")
            continue


        validate_history = np.load(model_name + '_validate_history.npy', allow_pickle=True).item()

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
