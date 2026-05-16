import sys

import pandas as pd

sys.path.append("..")


COLUMNS_REQUIRED = [
    "date",
    "product_id",
    "product_name",
    "quantity_sold",
    "price",
    "stock_after",
]


class DataValidationError(Exception):  # Custom exception for data validation errors
    pass


def load_data(file_path):
    """
    Load data from a CSV file.
    Args: file_path (str): The path to the CSV file.
    Returns: pd.DataFrame: The loaded data as a pandas DataFrame.

    """
    try:
        df = pd.read_csv(file_path)

        df = df.rename(
            columns={
                "Date": "date",
                "ProductID": "product_id",
                "ProductName": "product_name",
                "UnitsSold": "quantity_sold",
                "UnitPrice": "price",
                "StockQuantity": "stock_after",
            }
        )

        missing_columns = [col for col in COLUMNS_REQUIRED if col not in df.columns]

        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def remove_unwanted_columns(df, columns_to_remove: list):
    """
    Remove unwanted columns from the DataFrame.
    Args: df (pd.DataFrame): The input DataFrame. columns_to_remove (list): List of column names to remove.
    Returns: pd.DataFrame: The DataFrame with unwanted columns removed.
    """
    return df.drop(columns=columns_to_remove, errors="ignore")


def convert_to_datetime(df, date_column):
    """
    Convert specified columns to datetime format.
    Args: df (pd.DataFrame): The input DataFrame. date_colums: Column name to convert to datetime format.
    Returns: pd.DataFrame: The DataFrame with specified columns converted to datetime format.
    """
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    return df


def sort_by_date_and_productId(df, date_column, productId_column):
    """
    Sort the DataFrame by date and product ID.
    Args: df (pd.DataFrame):
    The input DataFrame. date_column: Column name for date. productId_column: Column name for product Id.
    Returns: pd.DataFrame: The sorted DataFrame.
    """
    return df.sort_values(by=[date_column, productId_column])


def drop_null_values(df):
    """
    Drop rows with null values from the DataFrame.
    Args: df (pd.DataFrame): The input DataFrame.
    Returns: pd.DataFrame: The DataFrame with null values dropped.
    """
    return df.dropna()


def clean_data(df):
    """
    Clean the input DataFrame by performing a series of transformations.
    Args: df (pd.DataFrame): The input DataFrame.
    Returns: pd.DataFrame: The cleaned DataFrame.
    """
    df = remove_unwanted_columns(df, columns_to_remove=[])
    df = convert_to_datetime(df, date_column="date")
    df = sort_by_date_and_productId(
        df, date_column="date", productId_column="product_id"
    )
    df = drop_null_values(df)
    return df
