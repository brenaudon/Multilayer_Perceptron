"""
This script trains a deep neural network using the provided dataset and configuration file.

It includes:
    - Data preprocessing
    - Model initialization
    - Training with backpropagation
    - Evaluation and saving of results

Dependencies:
    - os
    - tqdm
    - numpy
    - argparse
    - matplotlib.pyplot
    - cut_dataset.py
    - parse_config.py
    - metrics_functions.py
    - data_manipulation.py
    - activation_functions.py
    - optimization_functions.py
    - initialization_functions.py

@cvar SEED: Random seed for reproducibility.
@type SEED: int
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from parse_config import parse_config
from cut_dataset import stratified_split_csv
from metrics_functions import MetricFunctions
from data_manipulation import prepare_data_training
from optimization_functions import OptimizationFunction
from initialization_functions import InitializationFunction
from activation_functions import ActivationFunction, softmax


SEED = 420
np.random.seed(SEED)

def categorical_cross_entropy(y_true, y_pred):
    """
    Compute the categorical cross-entropy loss between true and predicted labels.

    @param y_true: The true labels.
    @type  y_true: np.ndarray
    @param y_pred: The predicted labels.
    @type  y_pred: np.ndarray

    @return: The categorical cross-entropy loss.
    @rtype:  float
    """
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Small epsilon to avoid log(0)
    return loss

def dropout_layer(A, dropout_rate):
    """
    Applies dropout to activations A with probability dropout_rate.

    @param A: The activations to apply dropout to.
    @type  A: np.ndarray
    @param dropout_rate: The probability of dropout.
    @type  dropout_rate: float

    @return: The activations with dropout applied.
    @rtype:  np.ndarray
    """
    dropout_mask = (np.random.rand(*A.shape) < dropout_rate) / (1 - dropout_rate)
    return A * dropout_mask

def initialization(dimensions, config):
    """
    Initialize the parameters of the deep neural network.

    @param dimensions: The dimensions of the neural network.
    @type  dimensions: list
    @param config: The configuration dictionary.
    @type  config: dict

    @return: The initialized parameters.
    @rtype:  dict
    """
    parameters = {}
    c_len = len(dimensions)

    # Random Initialization
    for c in range(1, c_len):
        layer = config.get('layer' + str(c))
        if layer is not None:
            init_funct = InitializationFunction(layer.get('initialization_function', 'random_normal'))
        else :
            init_funct = InitializationFunction('random_normal')
        parameters['W' + str(c)] = init_funct.function((dimensions[c], dimensions[c - 1]))
        parameters['b' + str(c)] = np.zeros((dimensions[c], 1))  # Zero bias initialization

    return parameters

def forward_propagation(X, parameters, config, training=True):
    """
    Perform forward propagation through the deep neural network.

    @param X: The input data.
    @type  X: np.ndarray
    @param parameters: The parameters of the neural network.
    @type  parameters: dict
    @param config: The configuration dictionary.
    @type  config: dict
    @param training: Whether we are training or predicting (to apply dropout or not).
    @type  training: bool

    @return: The activations of each layer.
    @rtype:  dict
    """
    activations = {'A0': X}
    c_len = len(parameters) // 2
    dropout_rate = config.get('dropout_rate', 0.0)  # Default 0%, no dropout

    for c in range(1, c_len + 1):
        Z = parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameters['b' + str(c)]
        activations['Z' + str(c)] = Z  # Store pre-activation Z
        if c == c_len:
            activations['A' + str(c)] = softmax(Z)
        else:
            layer = config.get('layer' + str(c))
            af = ActivationFunction(layer.get('activation', 'sigmoid') if layer else 'sigmoid')
            A = af.function(Z)

            if training:  # Apply dropout only during training
                A = dropout_layer(A, dropout_rate)

            activations['A' + str(c)] = A

    return activations

def back_propagation(y, parameters, activations, config):
    """
    Perform backpropagation through the deep neural network.

    @param y: The true labels.
    @type  y: np.ndarray
    @param parameters: The parameters of the neural network.
    @type  parameters: dict
    @param activations: The activations of each layer.
    @type  activations: dict
    @param config: The configuration dictionary.
    @type  config: dict

    @return: The gradients of the parameters.
    @rtype:  dict
    """
    m = y.shape[1]
    c_len = len(parameters) // 2
    gradients = {}

    dZ = activations['A' + str(c_len)] - y

    for c in reversed(range(1, c_len + 1)):
        gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        if c > 1:
            layer = config.get('layer' + str(c))
            af = ActivationFunction(layer.get('activation', 'sigmoid')) if layer else ActivationFunction('sigmoid')
            dA_prev = np.dot(parameters['W' + str(c)].T, dZ)
            dZ = dA_prev * af.derivative(activations['Z' + str(c - 1)])

    return gradients

def update(gradients, parameters, epoch, optimization_function: OptimizationFunction):
    """
    Update the parameters of the neural network using the gradients.

    @param gradients: The gradients of the parameters.
    @type  gradients: dict
    @param parameters: The parameters of the neural network.
    @type  parameters: dict
    @param epoch: The current epoch.
    @type  epoch: int
    @param optimization_function: The optimization function to use.
    @type  optimization_function: OptimizationFunction

    @return: The updated parameters.
    @rtype:  dict
    """
    return optimization_function.update(parameters, gradients, epoch)

def predict(X, parameters, config):
    """
    Predict the labels of the input data.

    @param X: The input data.
    @type  X: np.ndarray
    @param parameters: The parameters of the neural network.
    @type  parameters: dict
    @param config: The configuration dictionary.
    @type  config: dict

    @return: The predicted labels.
    @rtype:  np.ndarray
    """
    activations = forward_propagation(X, parameters, config, training=False)
    c_len = len(parameters) // 2
    Af = activations['A' + str(c_len)]
    return np.argmax(Af, axis=0)  # Returns 0 for 'B' and 1 for 'M'

def evaluate(X, y, epoch, history, parameters, config):
    """
    Evaluate the loss and metrics (chosen in config json) of the model on the input data.

    @param X: The input data.
    @type  X: np.ndarray
    @param y: The true labels.
    @type  y: np.ndarray
    @param epoch: The current epoch.
    @type  epoch: int
    @param history: The history of the model metrics.
    @type  history: dict
    @param parameters: The parameters of the neural network.
    @type  parameters: dict
    @param config: The configuration dictionary.
    @type  config: dict

    @return: The updated history of the model metrics.
    @rtype:  dict
    """
    max_index = len(parameters) // 2
    metrics = MetricFunctions(config.get('metrics', []))

    activations = forward_propagation(X, parameters, config, training=False)
    Af = activations[f'A{max_index}']
    history['loss'][epoch] = categorical_cross_entropy(y, Af)
    y_pred = predict(X, parameters, config)
    y_true = np.argmax(y, axis=0)
    for metric in config.get('metrics', []):
        history[metric][epoch] = (metrics.functions[metric](y_true.flatten(), y_pred.flatten()))
    return history

def save_model(model_name, parameters, training_history, validate_history):
    """
    Save the model parameters for future usage and training and validation history for future model comparisons.

    @param model_name: The name of the model.
    @type  model_name: str
    @param parameters: The parameters of the neural network.
    @type  parameters: dict
    @param training_history: The history of the model metrics during training on training dataset.
    @type  training_history: dict
    @param validate_history: The history of the model metrics during training on validation dataset.
    @type  validate_history: dict
    """
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, '../models', model_name)
    os.makedirs(model_path, exist_ok=True)
    #save the parameters
    np.save(f'{model_path}/{model_name}.npy', parameters)
    print(f"Model parameters saved in models/{model_name}/{model_name}.npy")

    # Save training and validation history for comparison
    np.save(f'{model_path}/{model_name}_training_history.npy', training_history)
    np.save(f'{model_path}/{model_name}_validate_history.npy', validate_history)
    print(f"Training and validation history saved in models/{model_name}/{model_name}_training_history.npy and models/{model_name}/{model_name}_validate_history.npy")

def plot_learning_curve(training_history, validate_history, config):
    """
    Plot the learning curve of the model during training and validation.

    @param training_history: The history of the model metrics during training on training dataset.
    @type  training_history: dict
    @param validate_history: The history of the model metrics during training on validation dataset.
    @type  validate_history: dict
    @param config: The configuration dictionary.
    @type  config: dict
    """
    metrics = config.get('metrics', [])  # Get metrics list, default to empty if None
    num_metrics = len(metrics)
    nb_cols = min(4, num_metrics + 1)  # Max 4 columns per row
    nb_rows = (num_metrics + 1 + nb_cols - 1) // nb_cols  # Compute the number of rows needed

    plt.figure(figsize=(nb_cols * 4, nb_rows * 3))
    plt.subplot(nb_rows, 4, 1)
    plt.plot(training_history['loss'][:], label='training loss')
    plt.plot(validate_history['loss'][:], 'orange', linestyle='--', label='validation loss')
    plt.legend()
    for j, metric in enumerate(config.get('metrics', [])):
        plt.subplot(nb_rows, 4, 2 + j)
        plt.plot(training_history[metric][:], label=f'training {metric}')
        plt.plot(validate_history[metric][:], 'orange', linestyle='--', label=f'validation {metric}')
        plt.legend()
    plt.show()

def deep_neural_network(X_train, y_train, X_validate, y_validate, config):
    """
    Train a deep neural network on the provided dataset.

    @param X_train: The training data.
    @type  X_train: np.ndarray
    @param y_train: The training labels.
    @type  y_train: np.ndarray
    @param X_validate: The validation data.
    @type  X_validate: np.ndarray
    @param y_validate: The validation labels.
    @type  y_validate: np.ndarray
    @param config: The configuration dictionary.
    @type  config: dict
    """
    model_name = config.get('model_name', 'model')
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 2000)
    early_stopping = True if config.get('early_stopping_patience') is not None else False
    patience = config.get('early_stopping_patience', 10)  # Early stopping patience
    best_val_loss = float('inf')
    patience_counter = 0

    # initialise parameters
    c = 1
    dimensions = []
    while config.get('layer' + str(c)) is not None:
        dimensions.append(config.get('layer' + str(c)).get('nb_neurons', 24))
        c += 1
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])

    print(f"Model dimensions: {dimensions}")

    parameters = initialization(dimensions, config)
    best_parameters = parameters.copy()

    # dictionary to store model metrics history
    metrics = config.get('metrics', [])
    training_history = {metric: np.zeros((epochs, 1)) for metric in metrics}
    training_history['loss'] = np.zeros((epochs, 1))
    validate_history = {metric: np.zeros((epochs, 1)) for metric in metrics}
    validate_history['loss'] = np.zeros((epochs, 1))

    # Initialize optimization function
    optimization_function = OptimizationFunction(config.get('optimization', 'gradient_descent'), config)

    c_len = len(parameters) // 2
    num_batches = X_train.shape[1] // batch_size

    # Training loop with mini-batch gradient descent and early stopping
    epoch_range = tqdm(range(epochs)) if config.get('display') is not None and config.get('display') == 'tqdm' else range(epochs)
    for epoch in epoch_range:
        # Shuffle dataset at the start of each epoch
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        y_train_shuffled = y_train[:, permutation]

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[:, start:end]
            y_batch = y_train_shuffled[:, start:end]

            activations = forward_propagation(X_batch, parameters, config)
            gradients = back_propagation(y_batch, parameters, activations, config)
            parameters = update(gradients, parameters, epoch, optimization_function)

        # Compute training loss and metrics at the end of each epoch
        training_history = evaluate(X_train, y_train, epoch, training_history, parameters, config)

        # Compute validation loss and metrics at the end of each epoch
        validate_history = evaluate(X_validate, y_validate, epoch, validate_history, parameters, config)

        if config.get('display') is not None and config.get('display') != 'tqdm':
            print(f'epoch {epoch+1}/{epochs} - training loss: {training_history[epoch, 0]:.4f} - validation loss: {validate_history[epoch, 0]:.4f}')

        # Early Stopping Check
        if early_stopping:
            if validate_history['loss'][epoch] < best_val_loss:
                best_val_loss = validate_history['loss'][epoch]
                patience_counter = 0
                best_parameters = parameters.copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}/{epochs}")
                    parameters = best_parameters
                    training_history['loss'] = training_history['loss'][:epoch]
                    validate_history['loss'] = validate_history['loss'][:epoch]
                    for metric in metrics:
                        training_history[metric] = training_history[metric][:epoch]
                        validate_history[metric] = validate_history[metric][:epoch]
                    break

    # Save model parameters and training and validate history for future model comparisons
    save_model(model_name, parameters, training_history, validate_history)

    # Plot learning curve
    plot_learning_curve(training_history, validate_history, config)

if __name__ == "__main__":
    """
    Main function to train a deep neural network on a dataset.
    
    The script takes the following arguments:
        -d/--dataset: Path to the dataset CSV file.
        -c/--config: Path to the configuration JSON file.
        -s/--save: Save the split dataset (optional).
    """
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset with a configuration file.")

    # Define command-line arguments
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration JSON file.")
    parser.add_argument("-s", "--save", type=bool, required=False, default=False, help="Save training and validate dataset (cut 90/10 of provided dataset).")

    # Parse arguments
    args = parser.parse_args()

    # Extract paths
    csv_file_path = args.dataset
    config_file_path = args.config
    save = args.save

    # Load the data from the CSV file and split it into training and validation sets
    df_train, df_validate = stratified_split_csv(csv_file_path, 90, save)

    # Prepare the data for training (PCA)
    df_train, eigenvectors, mean, std = prepare_data_training(df_train)
    df_validate, _, _, _ = prepare_data_training(df_validate, eigenvectors, mean, std)

    # Split the data into features and target
    X_train = df_train.drop(columns=['ID', 'Diagnosis']).T
    X_train = X_train.to_numpy()
    y_train = np.array(df_train['Diagnosis'].map({'M': [1, 0], 'B': [0, 1]}).tolist()).T
    X_validate = df_validate.drop(columns=['ID', 'Diagnosis']).T
    X_validate = X_validate.to_numpy()
    y_validate = np.array(df_validate['Diagnosis'].map({'M': [1, 0], 'B': [0, 1]}).tolist()).T

    print(f'X_train shape: {X_train.shape}')
    print(f'X_validate shape: {X_validate.shape}')

    # Load the configuration file
    config_dict = parse_config(config_file_path)

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Save config for use in predict script
    model_name = config_dict.get('model_name', 'model')
    model_path = os.path.join(script_dir, '../models', model_name)
    os.makedirs(model_path, exist_ok=True)
    np.save(f'{model_path}/{config_dict.get("model_name", "model")}_config.npy', config_dict)
    print(f"Configuration for this model saved in models/{config_dict.get('model_name', 'model')}/{config_dict.get('model_name', 'model')}_config.npy")

    # Save PCA parameters in a file for use in predict script
    np.savez(f'{model_path}/{model_name}_pca_parameters.npz', eigenvectors=eigenvectors, mean=mean, std=std)
    print(f"PCA parameters for data preparation for this model saved in models/{config_dict.get('model_name', 'model')}/{model_name}_pca_parameters.npz")

    deep_neural_network(X_train, y_train, X_validate, y_validate, config_dict)

    print("Training done!")