import pandas as pd


def split_data(df=pd.DataFrame):
    """
    Split the DataFrame into training and testing sets.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        tuple: A tuple containing the training, validation, and testing DataFrames.
    """
    train_size = 0.7
    val_size = 0.15

    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def split_features_and_target(df=pd.DataFrame):
    """
    Split the DataFrame into features and target variable.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        tuple: A tuple containing the features DataFrame and target DataFrame."""

    X = df.drop(columns=["quantity_sold"])
    y = df["quantity_sold"]
    return X, y
