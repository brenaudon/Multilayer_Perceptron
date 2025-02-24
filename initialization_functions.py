"""
This file provides various initialization functions for use in neural networks.

The script includes custom implementations for several initialization functions.

Dependencies:
    - numpy
"""

import numpy as np

class InitializationFunction:
    """
    A class to represent and dynamically call various initialization functions.

    @ivar name: The name of the initialization function.
    @type name: str
    @ivar function: The initialization function.
    @type function: function
    @ivar initialization_functions: Dictionary of available initialization functions.
    @type initialization_functions: dictionary
    """
    def __init__(self, name):
        """
        Initialize the InitializationFunction class with the given initialization function name.

        @param name: The name of the initialization function.
        @type  name: str
        """
        self.initialization_functions = {
            'random_normal': self.random_normal,
            'random_uniform': self.random_uniform,
            'he_normal': self.he_normal,
            'he_uniform': self.he_uniform,
            'xavier_glorot_normal': self.xavier_glorot_normal,
            'xavier_glorot_uniform': self.xavier_glorot_uniform,
            'lecun_normal': self.lecun_normal,
            'lecun_uniform': self.lecun_uniform,
            'selu': self.selu,
        }

        np.random.seed(10)  # Reproducibility
        self.name = name
        self.function = self.initialization_functions.get(name, self.unknown_initialization)

    def unknown_initialization(self, dimensions):
        """
        Raise an error for an unknown initialization function.
        """
        raise ValueError(f"Unknown initialization function: {self.name}")

    def random_normal(self, dimensions):
        """
        Random Normal initialization.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: Randomly initialized matrix from a normal distribution.
        @rtype: numpy.ndarray
        """
        return np.random.randn(*dimensions)

    def random_uniform(self, dimensions, minval=-1.0, maxval=1.0):
        """
        Random Uniform initialization.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple
        @param minval: Minimum value of the uniform distribution.
        @type  minval: float
        @param maxval: Maximum value of the uniform distribution.
        @type  maxval: float

        @return: Randomly initialized matrix from a uniform distribution.
        @rtype: numpy.ndarray
        """
        return np.random.uniform(low=minval, high=maxval, size=dimensions)

    def he_normal(self, dimensions):
        """
        He Normal initialization function.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: The He Normal initialized weight matrix.
        @rtype: numpy.ndarray
        """
        return np.random.randn(*dimensions) * np.sqrt(2. / dimensions[1])

    def he_uniform(self, dimensions):
        """
        He Uniform initialization function.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: The He Uniform initialized weight matrix.
        @rtype: numpy.ndarray
        """
        limit = np.sqrt(6 / dimensions[1])
        return np.random.uniform(-limit, limit, dimensions)

    def xavier_glorot_normal(self, dimensions):
        """
        Xavier Glorot Normal initialization function.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: The Xavier Glorot Normal initialized weight matrix.
        @rtype: numpy.ndarray
        """
        return np.random.randn(dimensions[0]) * np.sqrt(2. / (dimensions[0] + dimensions[1]))

    def xavier_glorot_uniform(self, dimensions):
        """
        Xavier Glorot Uniform initialization function.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: The Xavier Glorot Uniform initialized weight matrix.
        @rtype: numpy.ndarray
        """
        limit = np.sqrt(6 / (dimensions[0] + dimensions[1]))
        return np.random.uniform(-limit, limit, dimensions)

    def lecun_normal(self, dimensions):
        """
        Lecun Normal initialization function.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: The Lecun Normal initialized weight matrix.
        @rtype: numpy.ndarray
        """
        return np.random.randn(*dimensions) * np.sqrt(1. / dimensions[1])

    def lecun_uniform(self, dimensions):
        """
        Lecun Uniform initialization function.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: The Lecun Uniform initialized weight matrix.
        @rtype: numpy.ndarray
        """
        limit = np.sqrt(3 / dimensions[1])
        return np.random.uniform(-limit, limit, dimensions)

    def selu(self, dimensions):
        """
        Scaled Exponential Linear Unit (SELU) initialization function.

        @param dimensions: Shape of weight matrix (n_out, n_in).
        @type  dimensions: tuple

        @return: The SELU initialized weight matrix.
        @rtype: numpy.ndarray
        """
        scale = 1. / np.sqrt(dimensions[1])
        return np.random.randn(*dimensions) * scale

