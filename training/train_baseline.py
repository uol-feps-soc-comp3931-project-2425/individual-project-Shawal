import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

#load the preprocessed dataset
df = pd.read_csv("../model_datasets/Baseline_Model_Dataset.csv") 

#prepare features and target
y = df["CSAT_Score"]
X = df.drop(columns=["CSAT_Score"])

#train/test split method

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#train model
xgb_model_split = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=1
)

xgb_model_split.fit(X_train, y_train)

#evaluation
y_pred_split = xgb_model_split.predict(X_test)
rmse_split = np.sqrt(mean_squared_error(y_test, y_pred_split))
r2_split = r2_score(y_test, y_pred_split)

print(f"\nTrain/Test Split Results:")
print(f"RMSE: {rmse_split:.4f}")
print(f"R² Score: {r2_split:.4f}")

#save model and results
joblib.dump(xgb_model_split, "Baseline_XGBoost_Model_TrainTest.pkl")

#5-fold cross-validation method

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
r2_scores = []

for train_index, test_index in kfold.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    model_cv = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model_cv.fit(X_train_cv, y_train_cv)

    y_pred_cv = model_cv.predict(X_test_cv)

    rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
    r2 = r2_score(y_test_cv, y_pred_cv)

    rmse_scores.append(rmse)
    r2_scores.append(r2)

print(f"\n5-Fold Cross-Validation Results:")
print("RMSE per fold:", np.round(rmse_scores, 4))
print("Average RMSE:", round(np.mean(rmse_scores), 4))
print("R² per fold:", np.round(r2_scores, 4))
print("Average R²:", round(np.mean(r2_scores), 4))