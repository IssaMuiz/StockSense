import pandas as pd

num_features = [
    "price",
    "stock_after",
    "quantity_sold_lag1",
    "Day_of_week",
]

cat_features = ["product_id"]


def add_engineered_features(df: pd.DataFrame):
    df = df.copy()

    df = df.sort_values(["product_id", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["quantity_sold_lag1"] = df.groupby("product_id")["quantity_sold"].shift(1)

    df["Day_of_week"] = df["date"].dt.dayofweek

    df = df.drop(columns=["date", "product_name"], errors="ignore")
    df = df.dropna()

    return df
