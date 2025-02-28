"""
This file provides various optimization functions for use in neural networks.

The script includes custom implementations for several optimization functions.

Dependencies:
    - numpy
    - schedule_functions.py
"""

import numpy as np
from schedule_functions import ScheduleFunction

class OptimizationFunction:
    """
    A class to represent and dynamically call various optimization functions.

    @ivar optimizer_name: The name of the optimization function.
    @type optimizer_name: str
    @ivar config: The configuration dictionary for the neural network.
    @type config: dict
    @ivar function: The optimization function.
    @type function: function
    @ivar optimization_functions: Dictionary of available optimization functions.
    @type optimization_functions: dictionary
    """
    def __init__(self, optimizer_name, config):
        """
        Initialize the OptimizationFunction class with the given optimization function name.

        @param optimizer_name: The name of the optimization function.
        @type  optimizer_name: str
        """
        self.optimization_functions = {
            'gradient_descent': self.gradient_descent,
            # 'nesterov_momentum': self.nesterov_momentum,
            # 'rmsprop': self.rmsprop,
            # 'adam': self.adam,
            # 'adagrad': self.adagrad,
            # 'nadam': self.nadam,
            # 'adamw': self.adamw,
        }

        self.optimizer_name = optimizer_name
        self.config = config
        self.function = self.optimization_functions.get(optimizer_name, self.unknown_optimizer)

    def unknown_optimizer(self):
        """
        Raise an error for an unknown optimization function.
        """
        raise ValueError(f"Unknown optimization function: {self.optimizer_name}")

    def get_common_variables(self, config):
        """
        Get the variables used in the optimization function (common for all types).

        @param config: The configuration dictionary for the neural network.
        @type  config: dict

        @return: The variables for the optimization function.
        @rtype: dict
        """
        schedule_function = ScheduleFunction(self.config.get('schedule'), self.config) if self.config.get('schedule') else None
        initial_learning_rate = self.config.get('initial_learning_rate', 0.01) if schedule_function else self.config.get('learning_rate', 0.002)
        l1_lambda = self.config.get('l1_lambda', 0.0) # No L1 regularization by default
        l2_lambda = self.config.get('l2_lambda', 0.0) # No L2 regularization by default
        return schedule_function, initial_learning_rate, l1_lambda, l2_lambda

    def gradient_descent(self, gradients, parameters, epoch):
        """
        Gradient Descent optimization function.

        @param gradients: The gradients from backward_propagation.
        @type  gradients: dict
        @param parameters: The parameters from forward_propagation.
        @type  parameters: dict
        @param epoch: The current epoch.
        @type  epoch: int

        @return: The optimization function with the given learning rate.
        @rtype: function
        """
        schedule_function, initial_learning_rate, l1_lambda, l2_lambda = self.get_common_variables(self.config)
        c_len = len(parameters) // 2

        for c in range(1, c_len + 1):
            learning_rate = schedule_function.function(epoch) if schedule_function else initial_learning_rate
            parameters['W' + str(c)] -= learning_rate * (gradients['dW' + str(c)] + l1_lambda * np.sign(parameters['W' + str(c)]) + l2_lambda * parameters['W' + str(c)])
            parameters['b' + str(c)] -= learning_rate * gradients['db' + str(c)]

        return parameters

    def nesterov_momentum(self, gradients, parameters, epoch):
        schedule_function, initial_learning_rate, l1_lambda, l2_lambda = self.get_common_variables(self.config)
        beta = self.config.get('beta', 0.9)
        c_len = len(parameters) // 2
        velocity = {key: np.zeros_like(val) for key, val in parameters.items()}
        for c in range(1, c_len + 1):
            learning_rate = schedule_function.function(epoch) if schedule_function else initial_learning_rate
            velocity['dW' + str(c)] = beta * velocity.get('dW' + str(c), 0) + (1 - beta) * gradients['dW' + str(c)]
            velocity['db' + str(c)] = beta * velocity.get('db' + str(c), 0) + (1 - beta) * gradients['db' + str(c)]
            parameters['W' + str(c)] -= learning_rate * (velocity['dW' + str(c)] + l2_lambda * parameters['W' + str(c)] + l1_lambda * np.sign(parameters['W' + str(c)]))
            parameters['b' + str(c)] -= learning_rate * velocity['db' + str(c)]
        return parameters

