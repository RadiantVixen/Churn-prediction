# from classification import baseline
from parse import parse_data
from mlp import predict


if __name__ == "__main__":
    print("This is the main entry point of the program.")
    ids, X_train, X_test, y_train, y_test = parse_data("data/bank_data_train.csv")
    # baseline(X_train, y_train, X_test, y_test)
    predict(ids, X_train, X_test, y_train, y_test)
# 


