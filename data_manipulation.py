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

def get_most_correlated_features_to_remove(df: pd.DataFrame) -> list:
    """Find the most correlated features in the data and return a list of features to remove.

    @param df: The DataFrame containing the data.
    @type  df: pd.DataFrame

    @return: The list of features to remove because they have a high correlation with another feature.
    @rtype:  list
    """

    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['ID'])
    correlation_matrix = numeric_df.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    #make a list of every couple of features with a correlation higher than 0.95 (keep couples)
    most_correlated = upper_triangle[upper_triangle > 0.9].stack().index.tolist()

    to_remove = []

    for couple in most_correlated:
        if couple[0] not in to_remove and couple[1] not in to_remove:
            to_remove.append(couple[1])

    #save to_remove in a config file
    with open('config.txt', 'w') as f:
        for item in to_remove:
            f.write("%s\n" % item)

    return to_remove

def prepare_data_training(df: pandas.DataFrame, to_remove: list | None = None) -> pandas.DataFrame:
    # Count the number of columns
    num_features = df.shape[1]

    # Create an adequate number of features
    features = ['ID', 'Diagnosis']
    features += [f'feature_{i}' for i in range(num_features - 2)]

    # Assign the features as column names
    df.columns = features

    # Normalize the numeric features
    df = normalize_numeric_features(df)

    # Remove the most correlated features
    if to_remove is None:
        to_remove = get_most_correlated_features_to_remove(df)
    df = df.drop(columns=to_remove)

    return df, to_remove
