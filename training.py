import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import sys
from data_manipulation import prepare_data_training
from cut_dataset import split_csv
import argparse
from activation_functions import ActivationFunction, softmax

def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Small epsilon to avoid log(0)
    return loss

def initialisation(dimensions, initialization='random'):

    parameters = {}
    c_len = len(dimensions)

    # Randon seed for reproducibility
    np.random.seed(10)

    # Random Initialization
    for c in range(1, c_len):
        parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)

    # He Normal Initialization
    # for c in range(1, c_len):
    #     parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1]) * np.sqrt(2. / dimensions[c - 1])
    #     parameters['b' + str(c)] = np.zeros((dimensions[c], 1))  # Zero bias initialization

    # He Uniform Initialization
    # for c in range(1, c_len):
    #     limit = np.sqrt(6 / dimensions[c - 1])  # He Uniform formula
    #     parameters['W' + str(c)] = np.random.uniform(-limit, limit, (dimensions[c], dimensions[c - 1]))
    #     parameters['b' + str(c)] = np.zeros((dimensions[c], 1))  # Biases are usually initialized to 0

    return parameters

def forward_propagation(X, parameters, af=ActivationFunction('sigmoid')):

    activations = {'A0': X}

    c_len = len(parameters) // 2

    for c in range(1, c_len + 1):
        Z = parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameters['b' + str(c)]
        activations['Z' + str(c)] = Z  # Store pre-activation Z
        if c == c_len:
            activations['A' + str(c)] = softmax(Z)
        else:
            activations['A' + str(c)] = af.function(Z)
    return activations

def back_propagation(y, parameters, activations, af=ActivationFunction('sigmoid')):

    m = y.shape[1]
    c_len = len(parameters) // 2
    gradients = {}

    dZ = activations['A' + str(c_len)] - y

    for c in reversed(range(1, c_len + 1)):
        gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        if c > 1:
            dA_prev = np.dot(parameters['W' + str(c)].T, dZ)
            dZ = dA_prev * af.derivative(activations['Z' + str(c - 1)])

    return gradients

def update(gradients, parameters, learning_rate):

    c_len = len(parameters) // 2

    for c in range(1, c_len + 1):
        parameters['W' + str(c)] = parameters['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parameters['b' + str(c)] = parameters['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parameters

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    c_len = len(parameters) // 2
    Af = activations['A' + str(c_len)]
    return np.argmax(Af, axis=0)  # Returns 0 for 'B' and 1 for 'M'

def deep_neural_network(X_train, y_train, X_validate, y_validate, config):

    # initialise parameters
    dimensions = list(config.get('hidden_layers'))
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    parameters = initialisation(dimensions, config.get('initialization'))

    # numpy table to store training history
    training_history = np.zeros((int(config.get('n_iter')), 2))
    validate_history = np.zeros((int(config.get('n_iter')), 2))

    af = ActivationFunction(config.get('activation'))

    c_len = len(parameters) // 2

    # gradient descent
    for i in tqdm(range(config.get('n_iter'))):

        activations = forward_propagation(X_train, parameters, af)
        gradients = back_propagation(y_train, parameters, activations, af)
        parameters = update(gradients, parameters, config.get('learning_rate'))
        Af = activations['A' + str(c_len)]

        # log_loss and accuracy calcul
        training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))
        y_train_pred = predict(X_train, parameters)
        y_train_true = np.argmax(y_train, axis=0)
        training_history[i, 1] = (accuracy_score(y_train_true.flatten(), y_train_pred.flatten()))

        activations_validate = forward_propagation(X_validate, parameters)
        Af_validate = activations_validate['A' + str(c_len)]
        validate_history[i, 0] = (log_loss(y_validate.flatten(), Af_validate.flatten()))
        y_validate_pred = predict(X_validate, parameters)
        y_validate_true = np.argmax(y_validate, axis=0)
        validate_history[i, 1] = (accuracy_score(y_validate_true.flatten(), y_validate_pred.flatten()))


    #save the parameters
    np.save('parameters.npy', parameters)

    # Plot learning curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='training loss')
    plt.plot(validate_history[:, 0], 'orange', linestyle='--', label='validation loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='training acc')
    plt.plot(validate_history[:, 1], 'orange', linestyle='--', label='validation acc')
    plt.legend()
    plt.show()

    return training_history

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--layers', type=int, nargs='+', required=True, help='Number of neurons in each hidden layer')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--loss', type=str, required=True, help='Loss function')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    return parser.parse_args()

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
        'activation': 'sigmoid',
        'initialization': 'random',
        'optimisation': 'gradient_descent',
        'hidden_layers': (36, 36, 36),
        'learning_rate': 0.002,
        # 'batch_size': 32,
        # 'epochs': 15000,
        'n_iter': 15000,
        'loss': 'categorical_cross_entropy'
    }

    deep_neural_network(X_train, y_train, X_validate, y_validate, config)

    print("Training done!")




