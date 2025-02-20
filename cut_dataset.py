import pandas as pd
import sys

def split_csv(file_path, percentage, save = False):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=10).reset_index(drop=True)

    # Calculate the split index
    split_index = int(len(df) * (percentage / 100))

    # Split the DataFrame
    df1 = df[:split_index]
    df2 = df[split_index + 1:]

    # Save the split DataFrames to new CSV files
    if save:
        df1.to_csv('data_train.csv', index=False, header=False)
        df2.to_csv('data_validation.csv', index=False, header=False)

    return df1, df2

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cut_dataset.py <csv_file_path> <percentage>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    percent = float(sys.argv[2])

    if not (0 <= percent <= 100):
        print("Percentage must be between 0 and 100")
        sys.exit(1)

    split_csv(csv_file_path, percent, True)