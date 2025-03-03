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
    @ivar optimization_functions: Dictionary of available optimization functions.
    @type optimization_functions: dictionary
    """
    def __init__(self, optimizer_name, config):
        """
        Initialize the OptimizationFunction class with the given optimization function name.

        @param optimizer_name: The name of the optimization function.
        @type  optimizer_name: str
        @param config: The configuration dictionary for the neural network.
        @type  config: dict
        """
        self.optimization_functions = {
            'gradient_descent': GradientDescent,
            'adagrad': AdaGrad,
            'momentum': Momentum,
            'rmsprop': RMSprop,
            'adam': Adam,
            'adamw': AdamW,
        }

        self.optimizer_name = optimizer_name
        self.optimizer_class = self.optimization_functions.get(optimizer_name, self.unknown_optimizer)
        # self.optimizer = self.optimizer_class(config)

    def unknown_optimizer(self):
        """
        Raise an error for an unknown optimization function.
        """
        raise ValueError(f"Unknown optimization function: {self.optimizer_name}")

class Optimizer:
    """
    A class to represent various optimization functions.

    @ivar schedule_function: The learning rate schedule function.
    @type schedule_function: ScheduleFunction
    @ivar learning_rate: The initial learning rate.
    @type learning_rate: float
    @ivar l1_lambda: L1 regularization lambda parameter.
    @type l1_lambda: float
    @ivar l2_lambda: L2 regularization lambda parameter.
    @type l2_lambda: float
    """
    def __init__(self, config):
        """
        Base class for optimization functions.

        @param config: Configuration dictionary for the neural network.
        @type config: dict
        """
        self.schedule_function = ScheduleFunction(config.get('schedule'), config) if config.get('schedule') else None
        if self.schedule_function is None:
            self.learning_rate = config.get('learning_rate', 0.002)
        else:
            self.learning_rate = config.get('schedule_params').get('initial_learning_rate', 0.01) if config.get('schedule_params') else config.get('learning_rate', 0.01)
        self.l1_lambda = config.get('l1_lambda', 0.0) # No L1 regularization by default
        self.l2_lambda = config.get('l2_lambda', 0.0) # No L2 regularization by default

    def update(self, parameters, gradients, epoch):
        """
        Update function to be overridden by subclasses.

        @param parameters: Dictionary containing weights and biases.
        @type  parameters: dict
        @param gradients: Dictionary containing gradients.
        @type  gradients: dict
        @param epoch: Current epoch.
        @type  epoch: int
        @return: Updated parameters.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class GradientDescent(Optimizer):
    """
    A class to represent Gradient Descent optimization function.
    """
    def update(self, parameters, gradients, epoch):
        """
        Updates parameters using Batch Gradient Descent with possible learning rate schedule and L1/L2 regularization.

        @param parameters: Dictionary containing weights and biases.
        @type  parameters: dict
        @param gradients: Dictionary containing gradients.
        @type  gradients: dict
        @param epoch: Current epoch.
        @type  epoch: int
        @return: Updated parameters.
        """
        c_len = len(parameters) // 2

        for c in range(1, c_len + 1):
            learning_rate = self.schedule_function.function(epoch) if self.schedule_function else self.learning_rate
            parameters['W' + str(c)] -= learning_rate * (gradients['dW' + str(c)] + self.l1_lambda * np.sign(parameters['W' + str(c)]) + self.l2_lambda * parameters['W' + str(c)])
            parameters['b' + str(c)] -= learning_rate * gradients['db' + str(c)]

        return parameters

class AdaGrad(Optimizer):
    """
    A class to represent AdaGrad (Adaptive Gradient Algorithm) optimization function.

    @ivar G: Sum of past squared gradients.
    @type G: dict
    """
    def __init__(self, config):
        """
        AdaGrad (Adaptive Gradient Algorithm) optimizer.

        @param config: Configuration dictionary for the neural network.
        @type config: dict
        """
        super().__init__(config)
        self.G = {}

    def update(self, parameters, gradients, epoch):
        """
        Updates parameters using AdaGrad optimization with possible learning rate schedule and L1/L2 regularization.

        @param parameters: Dictionary containing weights and biases.
        @type  parameters: dict
        @param gradients: Dictionary containing gradients.
        @type  gradients: dict
        @param epoch: Current epoch.
        @type  epoch: int

        @return: Updated parameters.
        @rtype: dict
        """
        epsilon = 1e-8

        # Apply learning rate schedule if provided
        decayed_learning_rate = self.schedule_function.function(epoch) if self.schedule_function else self.learning_rate

        for key in parameters:
            if key not in self.G:
                self.G[key] = np.zeros_like(parameters[key])

            # Apply L1/L2 regularization to gradients before updating
            gradients[key] += self.l1_lambda * np.sign(parameters[key])  # L1 penalty
            gradients[key] += self.l2_lambda * parameters[key]  # L2 penalty

            # Accumulate squared gradients
            self.G[key] += parameters[key] ** 2

            # Update parameters
            parameters[key] -= (decayed_learning_rate / (np.sqrt(self.G[key]) + epsilon)) * gradients[key]

        return parameters

class Momentum(Optimizer):
    """
    A class to represent Momentum optimization function.

    @ivar beta: Momentum coefficient.
    @type beta: float
    @ivar velocity: Dictionary containing velocity for each parameter.
    @type velocity: dict
    """
    def __init__(self, config):
        """
        Momentum optimizer initialization.

        @param config: Configuration dictionary for the neural network.
        @type config: dict
        """
        super().__init__(config)
        self.beta = config.get('optimizer_params').get('beta', 0.9) if config.get('optimizer_params') else 0.9
        self.velocity = {}

    def update(self, parameters, gradients, epoch):
        """
        Updates parameters using Momentum optimization function with possible learning rate schedule and L1/L2 regularization.

        @param parameters: Dictionary containing weights and biases.
        @type  parameters: dict
        @param gradients: Dictionary containing gradients.
        @type  gradients: dict
        @param epoch: Current epoch.
        @type  epoch: int

        @return: Updated parameters.
        @rtype: dict
        """

        # Apply learning rate schedule if provided
        decayed_learning_rate = self.schedule_function.function(epoch) if self.schedule_function else self.learning_rate

        for key in parameters:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(parameters[key])

            # Apply L1/L2 regularization to gradients before updating velocity
            gradients[key] += self.l1_lambda * np.sign(parameters[key])  # L1 penalty
            gradients[key] += self.l2_lambda * parameters[key]  # L2 penalty

            # Update velocity
            self.velocity[key] = self.beta * self.velocity[key] + (1 - self.beta) * gradients[key]

            # Update parameters
            parameters[key] -= decayed_learning_rate * self.velocity[key]

        return parameters

class RMSprop(Optimizer):
    """
    A class to represent RMSprop optimization function.

    @ivar beta: Decay rate for past squared gradients.
    @type beta: float
    @ivar v: Dictionary containing moving average of squared gradients for each parameter.
    @type v: dict
    """
    def __init__(self, config):
        """
        RMSprop optimizer initialization.

        @param config: Configuration dictionary for the neural network.
        @type config: dict
        """
        super().__init__(config)
        self.beta = config.get('optimizer_params').get('beta', 0.9) if config.get('optimizer_params') else 0.9
        self.v = {}

    def update(self, parameters, gradients, epoch):
        """
        Updates parameters using RMSprop with possible learning rate schedule and L1/L2 regularization.

        @param parameters: Dictionary containing weights and biases.
        @type  parameters: dict
        @param gradients: Dictionary containing gradients.
        @type  gradients: dict
        @param epoch: Current epoch.
        @type  epoch: int

        @return: Updated parameters.
        @rtype: dict
        """

        epsilon = 1e-8

        # Apply learning rate schedule if provided
        decayed_learning_rate = self.schedule_function.function(epoch) if self.schedule_function else self.learning_rate

        for key in parameters:
            if key not in self.v:
                self.v[key] = np.zeros_like(parameters[key])

            # Apply L1/L2 regularization to gradients before updating velocity
            gradients[key] += self.l1_lambda * np.sign(parameters[key])  # L1 penalty
            gradients[key] += self.l2_lambda * parameters[key]  # L2 penalty

            # Update moving average of squared gradients
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (gradients[key] ** 2)

            # Apply RMSprop update
            parameters[key] -= (decayed_learning_rate / (np.sqrt(self.v[key]) + epsilon)) * gradients[key]

        return parameters

class Adam(Optimizer):
    """
    A class to represent Adam optimization function.

    @ivar beta1: Decay rate for the first moment estimate.
    @type beta1: float
    @ivar beta2: Decay rate for the second moment estimate.
    @type beta2: float
    @ivar m: Dictionary containing first moment estimate (Exponentially decaying average of past gradients) for each parameter.
    @type m: dict
    @ivar v: Dictionary containing second moment estimate (Exponentially decaying average of past squared gradients) for each parameter.
    @type v: dict
    @ivar t: Current timestep.
    @type t: int
    """
    def __init__(self, config):
        """
        Adam optimizer initialization.

        @param config: Configuration dictionary for the neural network.
        @type config: dict
        """
        super().__init__(config)
        self.beta1 = config.get('optimizer_params').get('beta1', 0.9) if config.get('optimizer_params') else 0.9
        self.beta2 = config.get('optimizer_params').get('beta2', 0.999) if config.get('optimizer_params') else 0.999
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, parameters, gradients, epoch):
        """
        Updates parameters using Adam with possible learning rate schedule and L1/L2 regularization.

        @param parameters: Dictionary containing weights and biases.
        @type  parameters: dict
        @param gradients: Dictionary containing gradients.
        @type  gradients: dict
        @param epoch: Current epoch.
        @type  epoch: int

        @return: Updated parameters.
        @rtype: dict
        """
        self.t += 1
        epsilon = 1e-8

        # Apply learning rate schedule if provided
        decayed_learning_rate = self.schedule_function.function(epoch) if self.schedule_function else self.learning_rate

        for key in parameters:
            if key not in self.m:
                self.m[key] = np.zeros_like(parameters[key])
                self.v[key] = np.zeros_like(parameters[key])

            # Apply L1/L2 regularization to gradients before updating moments
            gradients[key] += self.l1_lambda * np.sign(parameters[key])  # L1 penalty
            gradients[key] += self.l2_lambda * parameters[key]  # L2 penalty

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            parameters[key] -= decayed_learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        return parameters

class AdamW(Adam):
    """
    A class to represent AdamW optimization function.

    @ivar weight_decay: Weight decay rate.
    @type weight_decay: float
    """
    def __init__(self, config):
        """
        AdamW Optimizer (Adam with Decoupled Weight Decay). Inherits from Adam and applies weight decay separately.

        @param config: Configuration dictionary for the neural network.
        @type config: dict
        """
        super().__init__(config)
        self.weight_decay = config.get('optimizer_params').get('weight_decay', 0.01) if config.get('optimizer_params') else 0.01

    def update(self, parameters, gradients, epoch):
        """
        Updates parameters using AdamW optimization with possible learning rate schedule and L1/L2 regularization (should not use L2 regularization with it, see README.md).

        @param parameters: Dictionary containing weights and biases.
        @type  parameters: dict
        @param gradients: Dictionary containing gradients.
        @type  gradients: dict
        @param epoch: Current epoch.
        @type  epoch: int

        @return: Updated parameters.
        @rtype: dict
        """

        # Apply learning rate schedule if provided
        decayed_learning_rate = self.schedule_function.function(epoch) if self.schedule_function else self.learning_rate

        # Call the original Adam update
        parameters_adam = super().update(parameters, gradients, epoch)

        # Apply decoupled weight decay
        for key in parameters:
            parameters[key] = parameters_adam[key] - decayed_learning_rate * self.weight_decay * parameters[key]

        return parameters



