
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

def parse_data(file_path = "data/bank_data_train.csv"):
    df = pd.read_csv(file_path)

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())

    features = df.drop(columns=['TARGET'])
    scaler = StandardScaler()
    numeric_features = features.select_dtypes(include=['number'])

    X_standardized = scaler.fit_transform(numeric_features)

    X = X_standardized
    y = df[['TARGET']]

    print("Features: ", len(X))
    print("Labels: ", len(y))


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Total samples after cleaning: {len(df)}")

    return df[['ID']], X_train, X_test, y_train, y_test

    # print("Train samples: ", X_train)
    # print("Test samples: ", X_test)
    # print("Test samples: ", y_test)
    # print("Test samples: ", y_train)

    # train_df = pd.concat([X_train, y_train], axis=1)
    # test_df  = pd.concat([X_test, y_test], axis=1)

    # train_df.to_csv('train_set.csv', index=False)
    # test_df.to_csv('test_set.csv', index=False)

