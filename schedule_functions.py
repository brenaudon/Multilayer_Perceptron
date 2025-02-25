"""
This file provides various learning rate schedule functions for use in neural networks.

The script includes custom implementations for several learning rate schedule functions.

Dependencies:
    - numpy
"""

import numpy as np

class ScheduleFunction:
    """
    A class to represent and dynamically call various learning rate schedule functions.

    @ivar name: The name of the learning rate schedule function.
    @type name: str
    @ivar function: The learning rate schedule function.
    @type function: function
    @ivar schedule_functions: Dictionary of available learning rate schedule functions.
    @type schedule_functions: dictionary
    """
    def __init__(self, name):
        """
        Initialize the ScheduleFunction class with the given learning rate schedule function name.

        @param name: The name of the learning rate schedule function.
        @type  name: str
        """
        self.schedule_functions = {
            'step': self.step_decay,
            'exponential': self.exponential_decay,
            'cosine': self.cosine_annealing,
        }

        self.name = name
        self.function = self.schedule_functions.get(name, self.unknown_schedule)

    def unknown_schedule(self, dimensions):
        """
        Raise an error for an unknown learning rate schedule function.
        """
        raise ValueError(f"Unknown learning rate schedule function: {self.name}")

    def step_decay(self, epoch, initial_lr=0.01, drop_factor=0.1, epochs_drop=10):
        """
        Step decay learning rate schedule.

        @param epoch: The current epoch.
        @type  epoch: int
        @param initial_lr: The initial learning rate.
        @type  initial_lr: float
        @param drop_factor: The factor to drop the learning rate.
        @type  drop_factor: float
        @param epochs_drop: The number of epochs to drop the learning rate.
        @type  epochs_drop: int

        @return: The learning rate for the current epoch.
        @rtype: float
        """
        return initial_lr * (drop_factor ** (epoch // epochs_drop))

    def exponential_decay(self, epoch, initial_lr=0.01, decay_rate=0.01):
        """
        Exponential decay learning rate schedule.

        @param epoch: The current epoch.
        @type  epoch: int
        @param initial_lr: The initial learning rate.
        @type  initial_lr: float
        @param decay_rate: The decay rate.
        @type  decay_rate: float

        @return: The learning rate for the current epoch.
        @rtype: float
        """
        return initial_lr * np.exp(-decay_rate * epoch)

    def cosine_annealing(self, epoch, initial_lr=0.01, total_epochs=100):
        """
        Cosine Annealing decay learning rate schedule.

        @param epoch: The current epoch.
        @type  epoch: int
        @param initial_lr: The initial learning rate.
        @type  initial_lr: float
        @param total_epochs: The total number of epochs.
        @type  total_epochs: int

        @return: The learning rate for the current epoch.
        @rtype: float
        """
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

