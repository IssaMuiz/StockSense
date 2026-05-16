import sys

from pathlib import Path
from pipeline.features import add_engineered_features
from src.data_preprocessing import clean_data, load_data
from src.split_data import split_data, split_features_and_target

sys.path.append("..")


# PATH
RAW_DATA_PATH = Path("data/raw/stocksense_dataset.csv")
PROCESS_DATA_DIR = Path("data/processed/")
PROCESS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load raw data
df = load_data(RAW_DATA_PATH)

# Clean data
df = clean_data(df)

df = add_engineered_features(df)

# Train, validation and test split
train_df, val_df, test_df = split_data(df)

# Split features and target for training set
X_train, y_train = split_features_and_target(train_df)
# Split features and target for validation set
X_val, y_val = split_features_and_target(val_df)
# Split features and target for test set
X_test, y_test = split_features_and_target(test_df)


X_train.to_csv(PROCESS_DATA_DIR / "X_train.csv", index=False)
y_train.to_csv(PROCESS_DATA_DIR / "y_train.csv", index=False)
X_val.to_csv(PROCESS_DATA_DIR / "X_val.csv", index=False)
y_val.to_csv(PROCESS_DATA_DIR / "y_val.csv", index=False)
X_test.to_csv(PROCESS_DATA_DIR / "X_test.csv", index=False)
y_test.to_csv(PROCESS_DATA_DIR / "y_test.csv", index=False)

print("Data cleaned and splitted successfully")
