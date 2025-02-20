"""
This file provides various activation functions and their derivatives for use in neural networks.

The script includes custom implementations for several activation functions and their derivatives.

Dependencies:
    - numpy
"""

import numpy as np

def softmax(Z):
    """
    Compute the softmax of the input Z. Used for output layer activation.

    @param Z: The input Z.
    @type  Z: np.ndarray

    @return: The softmax probabilities.
    @rtype:  np.ndarray
    """
    assert len(Z.shape) == 2


    s = np.max(Z, axis=0, keepdims=True)
    e_x = np.exp(Z - s)
    div = np.sum(e_x, axis=0, keepdims=True)
    return e_x / div

class ActivationFunction:
    """
    A class to represent and dynamically call various activation functions and their derivatives.

    @ivar name: The name of the activation function.
    @type name: str
    @ivar function: The activation function.
    @type function: function
    @ivar derivative: The derivative of the activation function.
    @type derivative: function
    """
    def __init__(self, name):
        """
        Initialize the ActivationFunction class with the given activation function name.

        @param name: The name of the activation function.
        @type  name: str
        """
        self.name = name
        self.function = self.get_function(name)
        self.derivative = self.get_derivative(name)

    def get_function(self, name):
        """
        Get the activation function based on the name.

        @param name: The name of the activation function.
        @type  name: str

        @return: The activation function.
        @rtype:  function
        """
        if name == 'sigmoid':
            return self.sigmoid
        elif name == 'tanh':
            return self.tanh
        elif name == 'relu':
            return self.relu
        elif name == 'leaky_relu':
            return self.leaky_relu
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def get_derivative(self, name):
        """
        Get the derivative of the activation function based on the name.

        @param name: The name of the activation function.
        @type  name: str

        @return: The derivative of the activation function.
        @rtype:  function
        """
        if name == 'sigmoid':
            return self.sigmoid_derivative
        elif name == 'tanh':
            return self.tanh_derivative
        elif name == 'relu':
            return self.relu_derivative
        elif name == 'leaky_relu':
            return self.leaky_relu_derivative
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def sigmoid(self, Z):
        """
        Compute the sigmoid of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The sigmoid of Z.
        @rtype:  np.ndarray
        """
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        """
        Compute the derivative of the sigmoid function.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The derivative of the sigmoid function.
        @rtype:  np.ndarray
        """
        return Z * (1 - Z)

    def tanh(self, Z):
        """
        Compute the tanh of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The tanh of Z.
        @rtype:  np.ndarray
        """
        return np.exp(Z) - np.exp(-Z) / np.exp(Z) + np.exp(-Z)

    def tanh_derivative(self, Z):
        """
        Compute the derivative of the tanh function.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The derivative of the tanh function.
        @rtype:  np.ndarray
        """
        return 1 - Z ** 2

    def relu(self, Z):
        """
        Compute the ReLU of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The ReLU of Z.
        @rtype:  np.ndarray
        """
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """
        Compute the derivative of the ReLU function.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The derivative of the ReLU function.
        @rtype:  np.ndarray
        """
        return (Z > 0).astype(float)

    def leaky_relu(self, Z):
        """
        Compute the leaky ReLU of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The leaky ReLU of Z.
        @rtype:  np.ndarray
        """
        return np.maximum(0.01 * Z, Z)

    def leaky_relu_derivative(self, Z):
        """
        Compute the derivative of the leaky ReLU function.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The derivative of the leaky ReLU function.
        @rtype:  np.ndarray
        """
        return np.where(Z > 0, 1, 0.01)