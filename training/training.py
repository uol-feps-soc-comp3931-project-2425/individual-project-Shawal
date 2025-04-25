import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_model(dataset_path, model_name):
    print(f"\nTraining model for: {model_name}")
    
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["CSAT_Score"])
    y = df["CSAT_Score"]

    #Train/Test split eval
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    default_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    default_model.fit(X_train, y_train)
    y_pred = default_model.predict(X_test)
    print(f"Train/Test RMSE: {rmse(y_test, y_pred):.4f}")
    print(f"Train/Test R²: {r2_score(y_test, y_pred):.4f}")

    #5-Fold CV with default model
    print("\n5-Fold CV (Default Model)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores, r2_scores = [], []

    for train_idx, test_idx in kf.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        model_cv = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(X_test_cv)
        rmse_scores.append(rmse(y_test_cv, y_pred_cv))
        r2_scores.append(r2_score(y_test_cv, y_pred_cv))

    print("RMSE (per fold):", np.round(rmse_scores, 4))
    print("Avg RMSE:", round(np.mean(rmse_scores), 4))
    print("R² (per fold):", np.round(r2_scores, 4))
    print("Avg R²:", round(np.mean(r2_scores), 4))

    #Optuna Bayesian Optimisation
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "random_state": 42
        }

        model = XGBRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in kf.split(X):
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_test_cv)
            scores.append(rmse(y_test_cv, y_pred_cv))
        
        return np.mean(scores)

    print("\nRunning Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\nBest Trial:")
    print("Best RMSE:", round(study.best_value, 4))
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")

    #retrain with best params and save
    best_model = XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)

    y_pred_best = best_model.predict(X_test)
    r2_best = r2_score(y_test, y_pred_best)
    print("Best Tuned R²:", round(r2_best, 4))

    joblib.dump(best_model, f"{model_name}_XGBoost_Model.pkl")
    print(f"\nSaved model to: {model_name}_XGBoost_Model.pkl")

#run for all four models
train_model("../model_datasets/Baseline_Model_Dataset.csv", "Baseline")
train_model("../model_datasets/Sentiment_Model_Dataset.csv", "Sentiment")
train_model("../model_datasets/Clustering_Model_Dataset.csv", "Clustering")
train_model("../model_datasets/Combined_Model_Dataset.csv", "Combined")