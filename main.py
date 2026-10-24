import argparse
import pandas as pd
from classification import baseline, random_forest, mlp_classifier, keras_classifier, tensorflow_classifier
from parse import parse_data
from mlp import predict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn prediction experiments")
    parser.add_argument("--data", default="data/bank_data_train.csv")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Rows for a quick test; use 0 for the complete dataset")
    args = parser.parse_args()

    sample_size = None if args.sample_size == 0 else args.sample_size
    ids_train, ids_test, X_train, X_test, y_train, y_test = parse_data(args.data, sample_size)
    results = []
    # for name, runner in [("Baseline", baseline), ("Random forest", random_forest), ("Scikit-learn MLP", mlp_classifier)]:
    #     _, metrics = runner(X_train, y_train, X_test, y_test)
    #     results.append({"library": "scikit-learn", "algorithm": name, **metrics})

    # for name, runner in [("Keras MLP", keras_classifier), ("TensorFlow MLP", tensorflow_classifier)]:
    #     _, metrics = runner(X_train, y_train, X_test, y_test)
    #     results.append({"library": "TensorFlow", "algorithm": name, **metrics})

    metrics = predict(ids_test, X_train, X_test, y_train, y_test)
    results.append({"library": "NumPy", "algorithm": "OOP MLP", **metrics})
    print(pd.DataFrame(results).to_string(index=False))




