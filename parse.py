
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
    sample_size=None,
    random_state=42,
    preview_rows=5,):

    """Clean bank data, print a compact preview, and return stratified arrays."""
    df = pd.read_csv(file_path)
    required_columns = {"ID", "TARGET"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if preview_rows < 0:
        raise ValueError("preview_rows must be non-negative")

    preview_columns = list(df.columns[:8])
    for column in ("ID", "TARGET"):
        if column not in preview_columns:
            preview_columns.append(column)
    # print("Training data before cleaning:")
    # print(df.loc[:, preview_columns].head(preview_rows).to_string(index=False))
    df, duplicate_rows, duplicate_ids = _clean_dataframe(df)
    # print(
    #     f"\nRemoved {duplicate_rows} duplicate rows and "
    #     f"{duplicate_ids} duplicate IDs."
    # )

    if sample_size is not None:
        if sample_size < 10:
            raise ValueError("sample_size must be at least 10")
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)

    y = pd.to_numeric(df["TARGET"], errors="raise").astype(int)
    if y.nunique() < 2:
        raise ValueError("The selected data must contain both TARGET classes")

    DROP_FEATURES = {
    "LDEAL_ACT_DAYS_PCT_TR4",
    "LDEAL_ACT_DAYS_PCT_CURR",
    "CLNT_SALARY_VALUE",
    "LDEAL_YQZ_COM",
    "LDEAL_YQZ_CHRG",
    "AVG_PCT_MONTH_TO_PCLOSE",
    }
    
    features = _encode_features(df.drop(columns=["ID", "TARGET"]))
    # features = _encode_features(features.drop(columns=DROP_FEATURES))
    # print("\nTraining data after cleaning (encoded features):")
    # print(features.iloc[:, :8].head(preview_rows).to_string(index=False))
    # print(f"Rows: {len(features):,} | Features: {features.shape[1]}")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        features, y, df["ID"], test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (ids_train.reset_index(drop=True), ids_test.reset_index(drop=True),
            X_train, X_test, y_train.to_numpy(), y_test.to_numpy())


def parse_submission_data(train_path, test_path):
    """Prepare the labeled training data and unlabeled submission data together."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if "TARGET" not in train or "ID" not in train or "ID" not in test:
        raise ValueError("Both files must contain ID; the training file must contain TARGET")

    feature_columns = [column for column in train.columns if column not in {"ID", "TARGET"}]
    missing_columns = set(feature_columns).difference(test.columns)
    if missing_columns:
        raise ValueError(f"Test file is missing columns: {sorted(missing_columns)}")
    train, _, _ = _clean_dataframe(train)
    test, _, _ = _clean_dataframe(test)
    combined = pd.concat([train[feature_columns], test[feature_columns]], ignore_index=True)
    features = _encode_features(combined)

    return (
        pd.to_numeric(train["TARGET"], errors="raise").astype(int).to_numpy(),
        features.iloc[: len(train)].to_numpy(),
        features.iloc[len(train):].to_numpy(),
        test["ID"].reset_index(drop=True),
    )
