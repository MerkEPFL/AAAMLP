import joblib
import os
import pandas as pd
from sklearn import metrics
from sklearn import tree
import numpy as np


def run(fold):
    print(os.getcwd())
    # Read training data with folds
    df = pd.read_csv("./mnist_classification/input/train_folds.csv")

    # Train data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Validation data is the same as the fold
    df_val = df[df.kfold == fold].reset_index(drop=True)

    # Drop label column and convert into numpy array using .values to create train
    x_train = df_train.drop("label", axis=1).values

    # target is the label column
    y_train = df_train.label.values

    # Same for validation
    x_val = df_val.drop("label", axis=1).values
    y_val = df_val.label.values

    # Initialize decision tree classifier
    clf = tree.DecisionTreeClassifier()

    # Fit model on training data
    clf.fit(x_train, y_train)

    # Create predictions
    y_pred = clf.predict(x_val)

    # Calculate and print F1 score
    f1 = metrics.f1_score(y_val, y_pred, average='weighted')
    print(f"Fold={fold}, F1={f1}")

    # Save model
    joblib.dump(clf, f"./mnist_classification/models/dt_{fold}.bin")


if __name__ == "__main__":
    for f in np.arange(0, 10):
        run(fold=f)