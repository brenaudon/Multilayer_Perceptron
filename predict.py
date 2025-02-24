"""
This script performs predictions using a trained neural network model and give evaluation measures.

The script includes functions for making predictions and calculating binary cross-entropy loss.

Dependencies:
    - numpy
    - sklearn
    - pandas
    - sys
    - training.py
    - data_manipulation.py
"""

import numpy as np
import pandas as pd
import sys
from training import forward_propagation
from data_manipulation import prepare_data_training
from metrics_functions import MetricFunctions


def predict(X, parameters):
    """
    Perform prediction using the provided input data and model parameters.

    @param X: The input data.
    @type  X: np.ndarray
    @param parameters: The model parameters.
    @type  parameters: dict

    @return: The predicted probabilities and the predicted classes.
    @rtype:  tuple(np.ndarray, np.ndarray)
    """
    activations = forward_propagation(X, parameters)
    c_len = len(parameters) // 2
    probabilities = activations['A' + str(c_len)]  # Softmax or Sigmoid outputs
    return probabilities, np.argmax(probabilities, axis=0)  # Return the class with the highest probability


def binary_cross_entropy(y_true, y_pred_probs, epsilon=1e-12):
    """
    Compute the binary cross-entropy loss.

    @param y_true: The true labels.
    @type  y_true: np.ndarray
    @param y_pred_probs: The predicted probabilities.
    @type  y_pred_probs: np.ndarray
    @param epsilon: A small value to avoid log(0) issues.
    @type  epsilon: float

    @return: The binary cross-entropy loss.
    @rtype:  float
    """
    y_pred_probs = y_pred_probs[1, :]  # Take only P(M) if M=1, B=0
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)  # Avoid log(0) issues
    m = y_true.shape[1]
    loss = -1/m * np.sum(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
    return loss

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <data_csv_file_path> <model_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    model_file_path = sys.argv[2]

    # Load the data from the CSV file
    df = pd.read_csv(csv_file_path, header=None)

    # Open list of features to remove
    with open('to_remove.txt', 'r') as f:
        to_remove = f.read().splitlines()

    df, _ = prepare_data_training(df, to_remove)

    # Split the data into features and target
    X = df.drop(columns=['ID', 'Diagnosis']).T
    y = np.array(df['Diagnosis'].map({'M': [1, 0], 'B': [0, 1]}).tolist()).T

    # Check if Nan values are present in y
    if np.isnan(y).any():
        print("Error: NaN values are present in the target.")
        sys.exit(1)

    #Open the parameters file
    parameters = np.load('parameters.npy', allow_pickle=True).item()

    y_pred_probs, y_pred = predict(X, parameters)
    y_true = np.argmax(y, axis=0).T.reshape(1, -1)

    metrics = MetricFunctions(['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])

    for key in metrics.functions:
        print(f'{key}: {metrics.functions[key](y_true.flatten(), y_pred.flatten())}')

    print(f"Binary Cross-Entropy Loss: {binary_cross_entropy(y_true, y_pred_probs)}")
