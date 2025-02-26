import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from data_manipulation import prepare_data_training
from cut_dataset import split_csv
from activation_functions import ActivationFunction, softmax
from initialization_functions import InitializationFunction
from metrics_functions import MetricFunctions
from optimization_functions import OptimizationFunction

def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Small epsilon to avoid log(0)
    return loss

def dropout_layer(A, dropout_rate):
    """Applies dropout to activations A with probability dropout_rate."""
    dropout_mask = (np.random.rand(*A.shape) < dropout_rate) / (1 - dropout_rate)
    return A * dropout_mask

def initialisation(dimensions, config):
    parameters = {}
    c_len = len(dimensions)

    # Random Initialization
    for c in range(1, c_len):
        layer = config.get('layer' + str(c))
        if layer is not None:
            init_funct = InitializationFunction(layer.get('initialization', 'random_normal'))
        else :
            init_funct = InitializationFunction('random_normal')
        parameters['W' + str(c)] = init_funct.function((dimensions[c], dimensions[c - 1]))
        parameters['b' + str(c)] = np.zeros((dimensions[c], 1))  # Zero bias initialization

    return parameters

def forward_propagation(X, parameters, config, training=True):
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

def update(gradients, parameters, epoch, config):
    optimization_function = OptimizationFunction(config.get('optimisation', 'gradient_descent'), config)
    return optimization_function.function(gradients, parameters, epoch)

def predict(X, parameters, config):
    activations = forward_propagation(X, parameters, config, training=False)
    c_len = len(parameters) // 2
    Af = activations['A' + str(c_len)]
    return np.argmax(Af, axis=0)  # Returns 0 for 'B' and 1 for 'M'

def evaluate(X, y, epoch, history, parameters, config):
    max_index = len(parameters) // 2
    metrics = MetricFunctions(config.get('metrics'))

    activations = forward_propagation(X, parameters, config, training=False)
    Af = activations[f'A{max_index}']
    history[epoch, 0] = categorical_cross_entropy(y, Af)
    y_pred = predict(X, parameters, config)
    y_true = np.argmax(y, axis=0)
    for j, metric in enumerate(config.get('metrics', [])):
        history[epoch, 1 + j] = (metrics.functions[metric](y_true.flatten(), y_pred.flatten()))
    return history

def save_model(model_name, parameters, training_history, validate_history):
    #save the parameters
    np.save(f'{model_name}.npy', parameters)

    # Save training and validation history for comparison
    np.save(f'{model_name}_training_history.npy', training_history)
    np.save(f'{model_name}_validate_history.npy', validate_history)

def plot_learning_curve(training_history, validate_history, config):
    metrics = config.get('metrics', [])  # Get metrics list, default to empty if None
    num_metrics = len(metrics)
    nb_cols = min(4, num_metrics + 1)  # Max 4 columns per row
    nb_rows = (num_metrics + 1 + nb_cols - 1) // nb_cols  # Compute the number of rows needed

    plt.figure(figsize=(nb_cols * 4, nb_rows * 3))
    plt.subplot(nb_rows, 4, 1)
    plt.plot(training_history[:, 0], label='training loss')
    plt.plot(validate_history[:, 0], 'orange', linestyle='--', label='validation loss')
    plt.legend()
    for j, metric in enumerate(config.get('metrics')):
        plt.subplot(nb_rows, 4, 2 + j)
        plt.plot(training_history[:, 1 + j], label=f'training {metric}')
        plt.plot(validate_history[:, 1 + j], 'orange', linestyle='--', label=f'validation {metric}')
        plt.legend()
    plt.show()

def deep_neural_network(X_train, y_train, X_validate, y_validate, config):
    model_name = config.get('model_name', 'model')
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 50)
    patience = config.get('patience', 10)  # Early stopping patience
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

    parameters = initialisation(dimensions, config)
    best_parameters = parameters.copy()

    # numpy table to store model metrics history
    training_history = np.zeros((epochs, len(config.get('metrics')) + 1))
    validate_history = np.zeros((epochs, len(config.get('metrics')) + 1))

    c_len = len(parameters) // 2
    num_batches = X_train.shape[1] // batch_size

    # Training loop with mini-batch gradient descent and early stopping
    for epoch in tqdm(range(epochs)):
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
            parameters = update(gradients, parameters, epoch, config)

        # Compute training loss and metrics at the end of each epoch
        training_history = evaluate(X_train, y_train, epoch, training_history, parameters, config)

        # Compute validation loss and metrics at the end of each epoch
        validate_history = evaluate(X_validate, y_validate, epoch, validate_history, parameters, config)

        # Early Stopping Check
        if validate_history[epoch, 0] < best_val_loss:
            best_val_loss = validate_history[epoch, 0]
            patience_counter = 0
            best_parameters = parameters.copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                parameters = best_parameters
                training_history = training_history[:epoch, :]
                validate_history = validate_history[:epoch, :]
                break

    # Save model parameters and training and validate history for future model comparisons
    save_model(model_name, parameters, training_history, validate_history)

    # Plot learning curve
    plot_learning_curve(training_history, validate_history, config)

    return training_history

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python training.py <data_csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # Load the data from the CSV file
    df_train, df_validate = split_csv(csv_file_path, 90, True)

    df_train, eigenvectors, mean, std = prepare_data_training(df_train)
    df_validate, _, _, _ = prepare_data_training(df_validate, eigenvectors, mean, std)

    # Split the data into features and target
    X_train = df_train.drop(columns=['ID', 'Diagnosis']).T
    X_train = X_train.to_numpy()
    y_train = np.array(df_train['Diagnosis'].map({'M': [1, 0], 'B': [0, 1]}).tolist()).T
    X_validate = df_validate.drop(columns=['ID', 'Diagnosis']).T
    X_validate = X_validate.to_numpy()
    y_validate = np.array(df_validate['Diagnosis'].map({'M': [1, 0], 'B': [0, 1]}).tolist()).T

    config_dict = {
        'layer1' : {
            'nb_neurons': 36,
            'activation': 'sigmoid',
            'initialization': 'random_normal',
        },
        'layer2': {
            'nb_neurons': 36,
            'activation': 'sigmoid',
            'initialization': 'random_normal',
        },
        'layer3': {
            'nb_neurons': 36,
            'activation': 'sigmoid',
            'initialization': 'random_normal',
        },
        'optimisation': 'gradient_descent',
        'learning_rate': 0.002,
        'batch_size': 8,
        'epochs': 1500,
        'patience': 8,
        'dropout_rate': 0.1,
        'l1_lambda': 0.0,
        'l2_lambda': 0.0,
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'],
        'model_name': 'model',
    }

    model_name = config_dict.get('model_name', 'model')
    # Save PCA parameters in a file for use in predict script
    np.savez(f'{model_name}_pca_parameters.npz', eigenvectors=eigenvectors, mean=mean, std=std)

    deep_neural_network(X_train, y_train, X_validate, y_validate, config_dict)

    print("Training done!")