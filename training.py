import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from data_manipulation import prepare_data_training
from cut_dataset import split_csv
from activation_functions import ActivationFunction, softmax
from initialization_functions import InitializationFunction
from metrics_functions import MetricFunctions

def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Small epsilon to avoid log(0)
    return loss

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

def forward_propagation(X, parameters, config):

    activations = {'A0': X}

    c_len = len(parameters) // 2

    for c in range(1, c_len + 1):
        Z = parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameters['b' + str(c)]
        activations['Z' + str(c)] = Z  # Store pre-activation Z
        if c == c_len:
            activations['A' + str(c)] = softmax(Z)
        else:
            layer = config.get('layer' + str(c))
            if layer is not None:
                 af = ActivationFunction(layer.get('activation', 'sigmoid'))
            else :
                af = ActivationFunction('sigmoid')
            activations['A' + str(c)] = af.function(Z)
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
            if layer is not None:
                af = ActivationFunction(layer.get('activation', 'sigmoid'))
            else :
                af = ActivationFunction('sigmoid')
            dA_prev = np.dot(parameters['W' + str(c)].T, dZ)
            dZ = dA_prev * af.derivative(activations['Z' + str(c - 1)])

    return gradients

def update(gradients, parameters, learning_rate):

    c_len = len(parameters) // 2

    for c in range(1, c_len + 1):
        parameters['W' + str(c)] = parameters['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parameters['b' + str(c)] = parameters['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parameters

def predict(X, parameters, config):
    activations = forward_propagation(X, parameters, config)
    c_len = len(parameters) // 2
    Af = activations['A' + str(c_len)]
    return np.argmax(Af, axis=0)  # Returns 0 for 'B' and 1 for 'M'

def deep_neural_network(X_train, y_train, X_validate, y_validate, config):

    # initialise parameters
    c = 1
    dimensions = []
    while config.get('layer' + str(c)) is not None:
        dimensions.append(config.get('layer' + str(c)).get('nb_neurons'))
        c += 1
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])

    parameters = initialisation(dimensions, config)

    # numpy table to store training history
    training_history = np.zeros((int(config.get('n_iter')), len(config.get('metrics')) + 1))
    validate_history = np.zeros((int(config.get('n_iter')), len(config.get('metrics')) + 1))

    c_len = len(parameters) // 2

    metrics = MetricFunctions(config.get('metrics'))

    # gradient descent
    for i in tqdm(range(config.get('n_iter'))):

        activations = forward_propagation(X_train, parameters, config)
        gradients = back_propagation(y_train, parameters, activations, config)
        parameters = update(gradients, parameters, config.get('learning_rate'))
        Af = activations['A' + str(c_len)]

        # log_loss and accuracy calcul
        training_history[i, 0] = (categorical_cross_entropy(y_train, Af))
        y_train_pred = predict(X_train, parameters, config)
        y_train_true = np.argmax(y_train, axis=0)
        for j, metric in enumerate(config.get('metrics')):
            training_history[i, 1 + j] = (metrics.functions[metric](y_train_true.flatten(), y_train_pred.flatten()))

        activations_validate = forward_propagation(X_validate, parameters, config)
        Af_validate = activations_validate['A' + str(c_len)]
        validate_history[i, 0] = (categorical_cross_entropy(y_validate, Af_validate))
        y_validate_pred = predict(X_validate, parameters, config)
        y_validate_true = np.argmax(y_validate, axis=0)
        for j, metric in enumerate(config.get('metrics')):
            validate_history[i, 1 + j] = (metrics.functions[metric](y_validate_true.flatten(), y_validate_pred.flatten()))


    #save the parameters
    np.save('parameters.npy', parameters)

    # Plot learning curve
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

    return training_history

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python training.py <data_csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # Load the data from the CSV file
    df_train, df_validate = split_csv(csv_file_path, 90, True)

    df_train, to_remove = prepare_data_training(df_train)
    df_validate, _ = prepare_data_training(df_validate, to_remove)

    # Split the data into features and target
    X_train = df_train.drop(columns=['ID', 'Diagnosis']).T
    y_train = np.array(df_train['Diagnosis'].map({'M': [1, 0], 'B': [0, 1]}).tolist()).T
    X_validate = df_validate.drop(columns=['ID', 'Diagnosis']).T
    y_validate = np.array(df_validate['Diagnosis'].map({'M': [1, 0], 'B': [0, 1]}).tolist()).T

    config = {
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
        'hidden_layers': (36, 36, 36),
        # 'batch_size': 32,
        # 'epochs': 15000,
        'n_iter': 15000,
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    }

    deep_neural_network(X_train, y_train, X_validate, y_validate, config)

    print("Training done!")




