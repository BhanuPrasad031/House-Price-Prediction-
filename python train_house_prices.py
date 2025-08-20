import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

# ABSOLUTE PATH TO YOUR CSV (raw string to handle backslashes)
DATA_PATH = r"C:\Users\Bhanu Prasad\ML project\House Price Prediction Dataset.csv"
TARGET = "Price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_OUT = "model.joblib"

def get_regressor(random_state=42):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            tree_method="hist",
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state
        )

def infer_column_types(df, target):
    feature_df = df.drop(columns=[target])
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_df.columns if c not in num_cols]
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols, use_scaler_for_num=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler_for_num:
        num_steps.append(("scaler", StandardScaler(with_mean=False)))

    from sklearn.pipeline import Pipeline as SkPipeline
    num_pipeline = SkPipeline(steps=num_steps)

    cat_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    model = get_regressor()
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    return pipe

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found:\n{DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        print(f"ERROR: Target '{TARGET}' not found. Columns: {list(df.columns)}")
        sys.exit(1)

    # Drop Id-like column to avoid leakage
    for id_like in ["Id", "ID", "id"]:
        if id_like in df.columns and id_like != TARGET:
            df = df.drop(columns=[id_like])

    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values per column:\n", df.isna().sum())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    num_cols, cat_cols = infer_column_types(df, TARGET)
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(num_cols, cat_cols, use_scaler_for_num=False)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae, rmse, r2 = evaluate(y_test, preds)
    print(f"Test MAE:  {mae:,.2f}")
    print(f"Test RMSE: {rmse:,.2f}")
    print(f"Test R2:   {r2:.4f}")

    dump(pipe, MODEL_OUT)
    print(f"Saved trained pipeline to {MODEL_OUT}")

if __name__ == "__main__":
    main()
