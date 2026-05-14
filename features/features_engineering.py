import sys
import pandas as pd

sys.path.append("..")


def lag_features(df, group_by_column, target_column):
    """
    Create Lag features for a specified target column grouped by a specified column.
    Args:
         df (pd.DataFrame): The input DataFrame.
         group_by_column: Column name to group by.
         target_column: Column name for which to create lag features.
    Returns: pd.DataFrame: The DataFrame with lag features added."""

    df[f"{target_column}_lag"] = df.groupby(group_by_column)[target_column].shift(1)
    return df


def day_of_week_feature(df, date_column):
    """
    Create a day of the week feature from a date column.
    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column: Column name containing date information.
    Returns: pd.DataFrame: The DataFrame with the day of the week feature added."""
    df["day_of_week"] = df[date_column].dt.dayofweek
    return df
