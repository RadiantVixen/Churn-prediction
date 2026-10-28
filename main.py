import argparse
import pandas as pd
from classification import baseline, random_forest, mlp_classifier, keras_classifier, tensorflow_classifier
from parse import parse_data, _clean_dataframe, _encode_features
from mlp import evaluate_numpy_mlp

def write_csv(path, model):
    test_df = pd.read_csv(path)
    test_ids = test_df["ID"] if "ID" in test_df.columns else range(len(test_df))
    cleaned_test, _, _ = _clean_dataframe(test_df)
    encoded_test = _encode_features(cleaned_test.drop(columns=["ID", "TARGET"], errors="ignore"))
    _, X_test = model.data_processing(encoded_test)
    probabilities = model.predict_proba(X_test)
    submission = pd.DataFrame({
        "ID": test_ids, 
        "TARGET": probabilities
    })
    submission.to_csv("predictions.csv", index=False)
    print("Predictions successfully saved to predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn prediction experiments")
    parser.add_argument("--data", default="data/bank_data_train.csv")
    parser.add_argument("--testdata", default="data/bank_data_test.csv")
    parser.add_argument("--epoch", type=int, default="1")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Rows for a quick test; use 0 for the complete dataset")
    args = parser.parse_args()

    sample_size = None if args.sample_size == 0 else args.sample_size
    ids_train, ids_test, X_train, X_test, y_train, y_test = parse_data(args.data, sample_size)
    results = []
    for name, runner in [("Baseline", baseline), ("Random forest", random_forest), ("Scikit-learn MLP", mlp_classifier)]:
        _, metrics = runner(X_train, y_train, X_test, y_test)
        results.append({"library": "scikit-learn", "algorithm": name, **metrics})

    for name, runner in [("Keras MLP", keras_classifier), ("TensorFlow MLP", tensorflow_classifier)]:
        _, metrics = runner(X_train, y_train, X_test, y_test)
        results.append({"library": "TensorFlow", "algorithm": name, **metrics})

    model, metrics = evaluate_numpy_mlp(ids_test, X_train, y_train, X_test, y_test)
    results.append({"library": "NumPy", "algorithm": "OOP MLP", **metrics})
    
    print(pd.DataFrame(results).to_string(index=False))
    write_csv(args.testdata, model)




