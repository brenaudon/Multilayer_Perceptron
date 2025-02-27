import pandas as pd
import sys

def stratified_split_csv(file_path, percentage, save=False):
    """
    Split a CSV file into training and validation sets while maintaining class balance.

    @param file_path: The path to the CSV file.
    @type  file_path: str
    @param percentage: The percentage of data to include in the training set.
    @type  percentage: float
    @param save: Whether to save the split DataFrames to new CSV files.
    @type  save: bool

    @return: The training and validation DataFrames.
    @rtype:  pd.DataFrame, pd.DataFrame
    """
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Assuming the first column contains labels (diagnosis: "M" or "B")
    label_column = df.iloc[:, 1]

    # Separate data by class
    df_B = df[label_column == "B"]  # Benign
    df_M = df[label_column == "M"]  # Malignant

    # Compute split sizes
    train_size_B = int(len(df_B) * (percentage / 100))
    train_size_M = int(len(df_M) * (percentage / 100))

    # Shuffle each class separately
    df_B = df_B.sample(frac=1, random_state=420).reset_index(drop=True)
    df_M = df_M.sample(frac=1, random_state=420).reset_index(drop=True)

    # Split each class separately
    df_train = pd.concat([df_B[:train_size_B], df_M[:train_size_M]], ignore_index=True)
    df_valid = pd.concat([df_B[train_size_B:], df_M[train_size_M:]], ignore_index=True)

    # Shuffle final datasets
    df_train = df_train.sample(frac=1, random_state=420).reset_index(drop=True)
    df_valid = df_valid.sample(frac=1, random_state=420).reset_index(drop=True)

    # Save the split DataFrames to new CSV files
    if save:
        df_train.to_csv('data_train.csv', index=False, header=False)
        df_valid.to_csv('data_validation.csv', index=False, header=False)

    return df_train, df_valid

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cut_dataset.py <csv_file_path> <percentage>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    percent = float(sys.argv[2])

    if not (0 <= percent <= 100):
        print("Percentage must be between 0 and 100")
        sys.exit(1)

    stratified_split_csv(csv_file_path, percent, True)