import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

#load the data
df = pd.read_csv("../model_datasets/Baseline_Model_Dataset.csv")
X = df.drop(columns=["CSAT_Score"])
y = df["CSAT_Score"]

#define RMSE scorer (negative for Optuna minimisation)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#optuna objective
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

    rmse_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_scores.append(rmse(y_test, y_pred))

    return np.mean(rmse_scores)  

#run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

#print best results
print("\nBest Trial:")
print("Best RMSE:", round(study.best_value, 4))
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

#retrain best model for R² evaluation
best_model = XGBRegressor(**study.best_params)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
r2_list = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    rmse_list.append(rmse(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

print("\nFinal Evaluation with Best Parameters:")
print("RMSE per fold:", np.round(rmse_list, 4))
print("Average RMSE:", round(np.mean(rmse_list), 4))

print("R² per fold:", np.round(r2_list, 4))
print("Average R²:", round(np.mean(r2_list), 4))
