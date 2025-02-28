"""
This file provides various metric functions for use in neural networks evaluations.

The script includes custom implementations for several metric functions.

Dependencies:
    - numpy
"""

import numpy as np

class MetricFunctions:
    """
    A class to represent and dynamically call various metric functions.

    @ivar names: List of names of metric functions.
    @type names: list
    @ivar functions: Dictionary of chosen metric functions.
    @type functions: dictionary
    @ivar metric_functions: Dictionary of available metric functions.
    @type metric_functions: dictionary
    """
    def __init__(self, names: list):
        """
        Initialize the metric function dynamically based on the given name.

        @param names: List of names of the metrics.
        @type names: list
        """
        self.metric_functions = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1_score,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc,
        }

        self.names = names
        self.functions = { name: self.metric_functions.get(name, self.unknown_metric) for name in names }

    def unknown_metric(self):
        raise ValueError(f"Unknown metric function(s) in {self.names}")

    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy score.

        @param y_true: True labels.
        @type y_true: np.ndarray
        @param y_pred: Predicted probabilities.
        @type y_pred: np.ndarray

        @return: Accuracy score.
        @rtype: float
        """
        y_pred = np.round(y_pred)  # Convert probabilities to binary labels
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred):
        """
        Compute the precision score.

        @param y_true: True labels.
        @type y_true: np.ndarray
        @param y_pred: Predicted probabilities.
        @type y_pred: np.ndarray

        @return: Precision score.
        @rtype: float
        """
        y_pred = np.round(y_pred)
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        return true_positive / (true_positive + false_positive + 1e-8)

    def recall(self, y_true, y_pred):
        """
        Compute the recall score.

        @param y_true: True labels.
        @type y_true: np.ndarray
        @param y_pred: Predicted probabilities.
        @type y_pred: np.ndarray

        @return: Recall score.
        @rtype: float
        """
        y_pred = np.round(y_pred)
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_negative = np.sum((y_pred == 0) & (y_true == 1))
        return true_positive / (true_positive + false_negative + 1e-8)

    def f1_score(self, y_true, y_pred):
        """
        Compute the F1 score.

        @param y_true: True labels.
        @type y_true: np.ndarray
        @param y_pred: Predicted probabilities.
        @type y_pred: np.ndarray

        @return: F1 score.
        @rtype: float
        """
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall + 1e-8)

    def roc_auc(self, y_true, y_pred):
        """
        Compute the ROC-AUC score.

        @param y_true: True labels.
        @type y_true: np.ndarray
        @param y_pred: Predicted probabilities.
        @type y_pred: np.ndarray

        @return: ROC-AUC score.
        @rtype: float
        """
        sorted_indices = np.argsort(y_pred)[::-1]  # Sort in descending order
        y_true_sorted = y_true[sorted_indices]

        positives = np.sum(y_true)  # Number of positive samples
        negatives = len(y_true) - positives # Number of negative samples

        tpr = np.cumsum(y_true_sorted) / positives  # True positive rate
        fpr = np.cumsum(1 - y_true_sorted) / negatives  # False positive rate

        return self.trapz(tpr, fpr)


    def pr_auc(self, y_true, y_pred):
        """
        Compute PR-AUC (Precision-Recall Area Under Curve).

        @param y_true: True labels.
        @type y_true: np.ndarray
        @param y_pred: Predicted probabilities.
        @type y_pred: np.ndarray

        @return: PR-AUC score.
        @rtype: float
        """
        sorted_indices = np.argsort(y_pred)[::-1]  # Sort in descending order
        y_true_sorted = y_true[sorted_indices]

        true_positive = np.cumsum(y_true_sorted)
        false_positive = np.cumsum(1 - y_true_sorted)

        precision = true_positive / (true_positive + false_positive + 1e-8)  # Avoid division by zero
        recall = true_positive / np.sum(y_true)  # Recall calculation

        return self.trapz(precision, recall)


    def trapz(self, y, x):
        """
        Compute the Area Under the Curve (AUC) using the Trapezoidal Rule. Used in ROC-AUC and PR-AUC calculations.

        @param y: Array of function values.
        @type y: np.ndarray
        @param x: Array of x-coordinates.
        @type x: np.ndarray

        @return: Approximated integral.
        @rtype: float
        """
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)
