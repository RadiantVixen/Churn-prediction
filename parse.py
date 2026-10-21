
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_data(file_path="data/bank_data_train.csv", sample_size=None, random_state=42):
    """Load numeric bank data and return aligned, stratified train/test arrays."""
    df = pd.read_csv(file_path)
    required_columns = {"ID", "TARGET"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if sample_size is not None:
        if sample_size < 10:
            raise ValueError("sample_size must be at least 10")
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)

    y = df["TARGET"].astype(int)
    if y.nunique() < 2:
        raise ValueError("The selected data must contain both TARGET classes")

    features = df.drop(columns=["ID", "TARGET"])
    numeric_features = features.select_dtypes(include="number")
    if numeric_features.empty:
        raise ValueError("The dataset does not contain numeric features")
    numeric_features = numeric_features.replace([float("inf"), -float("inf")], pd.NA)
    numeric_features = numeric_features.fillna(numeric_features.median()).fillna(0)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        numeric_features, y, df["ID"], test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (ids_train.reset_index(drop=True), ids_test.reset_index(drop=True),
            X_train, X_test, y_train.to_numpy(), y_test.to_numpy())

