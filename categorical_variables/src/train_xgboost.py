import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    
    # Load training data with folds
    df = pd.read_csv("./categorical_variables/input/train_folds.csv")

    # All columns are features except id, target and folds
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    # We saw in EDA that we have NaNs, let's fill them with "NONE"
    # Let's convert all the columns in string anyway they are all categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # With a tree model we use a labe endcoding and not one hot encoding
    for col in features:
        lbl = preprocessing.LabelEncoder()

        lbl.fit(df[col])

        df.loc[:, col] = lbl.transform(df[col])

    # Get training data using fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Get validation data using fold
    df_val = df[df.kfold == fold].reset_index(drop=True)

    # Get training data
    x_train = df_train[features].values

    # Get validation data
    x_val = df_val[features].values
    
    print("Checkpoint1")
    # Initialize model: Logistic regression
    model = xgb.XGBClassifier(
        n_job=3,
        max_depth=7,
        n_estimators=200
    )

    # Model fit on training (label encoded)
    model.fit(x_train, df_train.target.values)

    print("Checkpoint2")
    # Predict on validation data
    # We want to use AUC so we need the probabilites
    # We will use the probabilites of 1s
    valid_preds = model.predict_proba(x_val)[:, 1]

    # Get AUC score
    auc = metrics.roc_auc_score(df_val.target.values, valid_preds)

    # Print AUC
    print(f"Fold {fold}: AUC={auc}")



if __name__ == "__main__":
    # Run for each fold
    df_fold = pd.read_csv("./categorical_variables/input/train_folds.csv")
    folds = df_fold.kfold.unique()
    for fold in folds:
        run(fold=fold)