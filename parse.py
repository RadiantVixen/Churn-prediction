import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _clean_dataframe(df):
    """Normalize values and remove duplicate records before feature preparation."""
    cleaned = df.copy()
    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].replace(r"^\s*$", np.nan, regex=True)
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    duplicate_rows = before - len(cleaned)
    if "ID" in cleaned:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset="ID", keep="first")
        duplicate_ids = before - len(cleaned)
    else:
        duplicate_ids = 0
    return cleaned, duplicate_rows, duplicate_ids


def _encode_features(features):
    """Impute numeric values and encode the 20 most common values per category."""
    numeric_features = features.select_dtypes(include="number").copy()
    categorical_features = features.select_dtypes(exclude="number").copy()
    if numeric_features.empty and categorical_features.empty:
        raise ValueError("The dataset does not contain features")

    numeric_features = numeric_features.fillna(numeric_features.median()).fillna(0)
    encoded_categorical = {}
    for column in categorical_features:
        values = categorical_features[column].fillna("__MISSING__").astype(str)
        top_values = values.value_counts().head(20).index
        encoded_categorical[column] = (
            values.where(values.isin(top_values), "__OTHER__")
            .map({value: index for index, value in enumerate(top_values)})
            .fillna(-1)
        )

    if encoded_categorical:
        return pd.concat(
            [
                numeric_features,
                pd.DataFrame(encoded_categorical, index=features.index),
            ],
            axis=1,
        )
    return numeric_features


def parse_data(
    file_path="data/bank_data_train.csv",
    random_state=42):

    """Clean bank data, print a compact preview, and return stratified arrays."""
    df = pd.read_csv(file_path)
    required_columns = {"ID", "TARGET"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df, duplicate_rows, duplicate_ids = _clean_dataframe(df)

    y = pd.to_numeric(df["TARGET"], errors="raise").astype(int)
    if y.nunique() < 2:
        raise ValueError("The selected data must contain both TARGET classes")

    
    features = _encode_features(df.drop(columns=["ID", "TARGET"]))

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        features, y, df["ID"], test_size=0.2, random_state=random_state, stratify=y
    )

    return (ids_train.reset_index(drop=True), ids_test.reset_index(drop=True),
            X_train, X_test, y_train.to_numpy(), y_test.to_numpy())
