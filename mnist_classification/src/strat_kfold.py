import pandas as pd
import os
from sklearn import model_selection

if __name__ == "__main__":

    # Setting directory
    os.chdir("/Users/Samuele 1/Documents/GitHub/AAAMLP/mnist_classification/input")

    # Importing training data called train.csv in a CSV format
    df = pd.read_csv("train.csv")

    # Creation of a new column called kfold and filled with -1
    df["kfold"] = -1

    # Randomize rows of train
    df = df.sample(frac=1).reset_index(drop=True)

    # Fetch targets
    y = df.label.values

    # Initiate the stratified kfold class from model_selection module with 5 folds
    kf = model_selection.StratifiedKFold(n_splits=10)

    # Fill the new fold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold
    
    # Save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)