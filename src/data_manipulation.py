"""
This script provides functions for normalizing and processing dataset features.

It includes:
    - Normalization of numeric features
    - Standardization of data
    - Principal Component Analysis (PCA)
    - Data preparation for training

Dependencies:
    - pandas
    - numpy
    - sklearn.preprocessing (MinMaxScaler)
"""

import pandas as pd
import numpy as np

def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardize data to zero mean and unit variance.

    @param data: The data to standardize.
    @type  data: np.ndarray

    @return: The standardized data.
    @rtype:  np.ndarray
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def compute_pca(df, variance_threshold: float = 0.95):
    """
    Compute PCA (Principal component analysis) and return the transformation matrix (only used on training data then applied to validation set as well so we have the same number of features in both).

    @param df: The DataFrame containing the data.
    @type  df: pd.DataFrame
    @param variance_threshold: The threshold for the cumulative variance to retain.
    @type  variance_threshold: float

    @return: The eigenvectors, mean, and standard deviation of the data.
    @rtype:  np.ndarray, np.ndarray, np.ndarray
    """

    numeric_df = df.drop(columns=['ID', 'Diagnosis'])
    data = numeric_df.values

    # Standardize training data
    train_standardized = standardize_data(data)

    # Compute covariance matrix
    covariance_matrix = np.cov(train_standardized, rowvar=False)

    # Compute eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Compute explained variance ratio
    explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Select number of components to retain at least the variance_threshold
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Select the top eigenvectors
    top_eigenvectors = sorted_eigenvectors[:, :num_components]

    print(f"PCA retained {num_components} components, explaining {cumulative_variance[num_components-1]:.2%} variance.")

    return top_eigenvectors, np.mean(data, axis=0), np.std(data, axis=0)

def apply_pca(df, eigenvectors: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply learned PCA transformation to new data.

    @param df: The DataFrame containing the data.
    @type  df: pd.DataFrame
    @param eigenvectors: The eigenvectors learned from the training data.
    @type  eigenvectors: np.ndarray
    @param mean: The mean of the training data.
    @type  mean: np.ndarray
    @param std: The standard deviation of the training data.
    @type  std: np.ndarray

    @return: The PCA-transformed data.
    @rtype:  np.ndarray
    """
    numeric_df = df.drop(columns=['ID', 'Diagnosis']).values
    standardized_data = (numeric_df - mean) / std  # Standardize using training mean/std
    return np.dot(standardized_data, eigenvectors)

def prepare_data_training(df: pd.DataFrame, eigenvectors = None, mean = None, std = None):
    """
    Prepare data: normalize, remove correlated features, and apply PCA if needed.

    @param df: The DataFrame containing the data.
    @type  df: pd.DataFrame
    @param eigenvectors: The eigenvectors learned from the training data.
    @type  eigenvectors: np.ndarray
    @param mean: The mean of the training data.
    @type  mean: np.ndarray
    @param std: The standard deviation of the training data.
    @type  std: np.ndarray

    @return: The DataFrame with PCA-transformed data.
    @rtype:  pd.DataFrame
    """

    # Assign column names if missing
    df.columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(df.shape[1] - 2)]

    # Extract 'ID' and 'Diagnosis' before transformation
    id_col = df[['ID']]
    diagnosis_col = df[['Diagnosis']]

    # Compute PCA on training set only
    if eigenvectors is None or mean is None or std is None:
        eigenvectors, mean, std = compute_pca(df, variance_threshold=0.95)

    # Transform both training and validation sets
    pca_transformed = apply_pca(df, eigenvectors, mean, std)

    # Convert PCA result into a DataFrame
    pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])], index=df.index)

    # Concatenate 'ID' and 'Diagnosis' back with PCA-transformed data
    final_df = pd.concat([id_col, diagnosis_col, pca_df], axis=1)

    return final_df, eigenvectors, mean, std

