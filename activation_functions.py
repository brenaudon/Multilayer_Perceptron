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
    @ivar activation_functions: Dictionary of available activation functions.
    @type activation_functions: dictionary
    @ivar activation_derivatives: Dictionary of available derivative of an activation functions.
    @type activation_derivatives: dictionary
    """

    def __init__(self, name):
        """
        Initialize the ActivationFunction class with the given activation function name.

        @param name: The name of the activation function.
        @type  name: str
        """
        self.activation_functions = {
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'relu': self.relu,
            'leaky_relu': self.leaky_relu,
            'elu': self.elu,
            'selu': self.selu,
            'swish': self.swish,
            'gelu': self.gelu
        }
        self.activation_derivatives = {
            'sigmoid': self.sigmoid_derivative,
            'tanh': self.tanh_derivative,
            'relu': self.relu_derivative,
            'leaky_relu': self.leaky_relu_derivative,
            'elu': self.elu_derivative,
            'selu': self.selu_derivative,
            'swish': self.swish_derivative,
            'gelu': self.gelu_derivative
        }
        self.name = name
        self.function = self.activation_functions.get(name, self.unknown_activation)
        self.derivative = self.activation_derivatives.get(name, self.unknown_derivative)

    def unknown_activation(self):
        """
        Raise an error for an unknown activation function.
        """
        raise ValueError(f"Unknown activation function: {self.name}")

    def unknown_derivative(self):
        """
        Raise an error for an unknown derivative function.
        """
        raise ValueError(f"Unknown activation derivative: {self.name}")

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

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray

        @return: The derivative of the sigmoid function.
        @rtype:  np.ndarray
        """
        sigmoid_Z = 1 / (1 + np.exp(-Z))
        return sigmoid_Z * (1 - sigmoid_Z)

    def tanh(self, Z):
        """
        Compute the tanh of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The tanh of Z.
        @rtype:  np.ndarray
        """
        return np.tanh(Z)

    def tanh_derivative(self, Z):
        """
        Compute the derivative of the tanh function.

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray

        @return: The derivative of the tanh function.
        @rtype:  np.ndarray
        """
        tanh_Z = np.tanh(Z)
        return 1 - tanh_Z ** 2  # Correct derivative

    def relu(self, Z):
        """
        Compute the ReLU (Rectified Linear Unit) of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The ReLU of Z.
        @rtype:  np.ndarray
        """
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """
        Compute the derivative of the ReLU function.

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray

        @return: The derivative of the ReLU function.
        @rtype:  np.ndarray
        """
        return (Z > 0).astype(float)

    def leaky_relu(self, Z, alpha=0.01):
        """
        Compute the leaky ReLU of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray
        @param alpha: Small slope for negative values
        @type  alpha: float

        @return: The leaky ReLU of Z.
        @rtype:  np.ndarray
        """
        return np.maximum(alpha * Z, Z)

    def leaky_relu_derivative(self, Z, alpha=0.01):
        """
        Compute the derivative of the leaky ReLU function.

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray
        @param alpha: Small slope for negative values
        @type  alpha: float

        @return: The derivative of the leaky ReLU function.
        @rtype:  np.ndarray
        """
        return np.where(Z > 0, 1, alpha)

    def elu(self, Z, alpha=1.0):
        """
        Compute the ELU (Exponential Linear Unit) of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray
        @param alpha: Constant factor for negative values
        @type  alpha: float

        @return: The ELU of Z.
        @rtype:  np.ndarray
        """
        return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

    def elu_derivative(self, Z, alpha=1.0):
        """
        Compute the derivative of the ELU function.

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray
        @param alpha: Constant factor for negative values
        @type  alpha: float

        @return: The derivative of the ELU function.
        @rtype:  np.ndarray
        """
        return np.where(Z > 0, 1, self.elu(Z, alpha) + alpha)

    def selu(self, Z, alpha=1.67326, scale=1.0507):
        """
        Compute the SELU (Scaled Exponential Linear Unit) of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray
        @param alpha: Constant factor for negative values
        @type  alpha: float
        @param scale: Constant factor for scaling
        @type  scale: float

        @return: The SELU of Z.
        @rtype:  np.ndarray
        """
        return scale * np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

    def selu_derivative(self, Z, alpha=1.67326, scale=1.0507):
        """
        Compute the derivative of the SELU function.

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray
        @param alpha: Constant factor for negative values
        @type  alpha: float
        @param scale: Constant factor for scaling
        @type  scale: float

        @return: The derivative of the SELU function.
        @rtype:  np.ndarray
        """
        return scale * np.where(Z > 0, 1, self.elu(Z, alpha) + alpha)

    def swish(self, Z):
        """
        Compute the Swish (Self-Gated) of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The Swish of Z.
        @rtype:  np.ndarray
        """
        return Z / (1 + np.exp(-Z))

    def swish_derivative(self, Z):
        """
        Compute the derivative of the Swish function.

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray

        @return: The derivative of the Swish function.
        @rtype:  np.ndarray
        """
        swish_Z = self.swish(Z)
        return swish_Z + (1 - swish_Z) * self.sigmoid(Z)

    def gelu(self, Z):
        """
        Compute the GELU (Gaussian Error Linear Unit) of the input Z.

        @param Z: The input Z.
        @type  Z: np.ndarray

        @return: The GELU of Z.
        @rtype:  np.ndarray
        """
        return 0.5 * Z * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * Z**3)))

    def gelu_derivative(self, Z):
        """
        Compute the derivative of the GELU function.

        @param Z: The pre-activation value Z.
        @type  Z: np.ndarray

        @return: The derivative of the GELU function.
        @rtype:  np.ndarray
        """
        c = np.sqrt(2 / np.pi)
        return 0.5 * (1 + np.tanh(c * (Z + 0.044715 * Z**3))) + \
            0.5 * Z * (1 - np.tanh(c * (Z + 0.044715 * Z**3)) ** 2) * \
            (c * (1 + 3 * 0.044715 * Z**2))
