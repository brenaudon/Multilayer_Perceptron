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
    def __init__(self, name, initial_lr=0.01, **kwargs):
        """
        Initialize the ScheduleFunction class with the given learning rate schedule function name.

        @param name: The name of the learning rate schedule function.
        @type  name: str
        """
        self.schedule_functions = {
            'step': self.step_decay,
            'exponential': self.exponential_decay,
            'time_based': self.time_based_decay,
            'cosine': self.cosine_annealing,
        }

        self.name = name
        self.lr = initial_lr
        self.kwargs = kwargs
        self.function = self.schedule_functions.get(name, self.unknown_schedule)

    def unknown_schedule(self):
        """
        Raise an error for an unknown learning rate schedule function.
        """
        raise ValueError(f"Unknown learning rate schedule function: {self.name}")

    def step_decay(self, epoch):
        """
        Step decay learning rate schedule. Reduce LR by a factor (drop_factor) every 'epochs_drop' epochs.

        @param epoch: The current epoch.
        @type  epoch: int

        @return: The learning rate for the current epoch.
        @rtype: float
        """
        drop_factor = self.kwargs.get("drop_factor", 0.1)
        epochs_drop = self.kwargs.get("epochs_drop", 10)
        return self.lr * (drop_factor ** (epoch // epochs_drop))

    def exponential_decay(self, epoch):
        """
        Exponential decay learning rate schedule. Reduce LR exponentially: lr = lr0 * exp(-decay_rate * epoch).

        @param epoch: The current epoch.
        @type  epoch: int

        @return: The learning rate for the current epoch.
        @rtype: float
        """
        decay_rate = self.kwargs.get("decay_rate", 0.01)
        return self.lr * np.exp(-decay_rate * epoch)

    def time_based_decay(self, epoch):
        """
        Time based decay learning rate schedule. Reduce LR using time-based decay: lr = lr0 / (1 + decay_rate * epoch).

        @param epoch: The current epoch.
        @type  epoch: int

        @return: The learning rate for the current epoch.
        @rtype: float
        """
        decay_rate = self.kwargs.get("decay_rate", 0.01)
        return self.lr / (1 + decay_rate * epoch)

    def cosine_annealing(self, epoch):
        """
        Cosine Annealing decay learning rate schedule.

        @param epoch: The current epoch.
        @type  epoch: int

        @return: The learning rate for the current epoch.
        @rtype: float
        """
        total_epochs = self.kwargs.get("total_epochs", 100)
        return self.lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))