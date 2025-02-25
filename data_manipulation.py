import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize every numeric feature in the DataFrame.

    @param df: The DataFrame containing the data.
    @type  df: pd.DataFrame

    @return: The DataFrame with normalized numeric features.
    @rtype:  pd.DataFrame
    """
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['ID'])
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_array, columns=numeric_df.columns, index=df.index)

    # Replace the original numeric columns with the normalized ones
    df[numeric_df.columns] = normalized_df

    return df

def standardize_data(data: np.ndarray) -> np.ndarray:
    """Standardize data to zero mean and unit variance."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def compute_pca(df, variance_threshold: float = 0.95):
    """Compute PCA only on training data and return the transformation matrix."""

    numeric_df = df.drop(columns=['ID', 'Diagnosis'])
    data = numeric_df.values

    # Standardize training data
    train_standardized = standardize_data(numeric_df)

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
    """Apply learned PCA transformation to new data."""
    numeric_df = df.drop(columns=['ID', 'Diagnosis']).values
    standardized_data = (numeric_df - mean) / std  # Standardize using training mean/std
    return np.dot(standardized_data, eigenvectors)

def prepare_data_training(df: pd.DataFrame, eigenvectors = None, mean = None, std = None):
    """Prepare data: normalize, remove correlated features, and apply PCA if needed."""

    # Assign column names if missing
    df.columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(df.shape[1] - 2)]

    # Extract 'ID' and 'Diagnosis' before transformation
    id_col = df[['ID']]
    diagnosis_col = df[['Diagnosis']]

    # Normalize numeric features
    df = normalize_numeric_features(df)

    # Compute PCA on training set only
    if eigenvectors is None or mean is None or std is None:
        eigenvectors, mean, std = compute_pca(df, variance_threshold=0.95)
        # Save PCA parameters in a file for use in predict script
        np.savez('pca_parameters.npz', eigenvectors=eigenvectors, mean=mean, std=std)

    # Transform both training and validation sets
    pca_transformed = apply_pca(df, eigenvectors, mean, std)

    # Convert PCA result into a DataFrame
    pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])], index=df.index)

    # Concatenate 'ID' and 'Diagnosis' back with PCA-transformed data
    final_df = pd.concat([id_col, diagnosis_col, pca_df], axis=1)

    return final_df, eigenvectors, mean, std

