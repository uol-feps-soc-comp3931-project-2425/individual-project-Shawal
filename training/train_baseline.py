import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

#load the preprocessed dataset
df = pd.read_csv("../model_datasets/Baseline_Model_Dataset.csv") 

#prepare features and target
y = df["CSAT_Score"]  
#remove target from features
X = df.drop(columns=["CSAT_Score"]) 

#split dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#train XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=1
)

xgb_model.fit(X_train, y_train)

#make predictions and evaluate
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

results_df = pd.DataFrame({
    "Actual_CSAT": y_test.values,
    "Predicted_CSAT": y_pred
})

#round predictions to 2 decimal places
results_df["Predicted_CSAT"] = results_df["Predicted_CSAT"].round(2)

results_df.to_csv("Baseline_Actual_vs_Predicted.csv", index=False)

print(f"RMSE: {rmse:.4f}")

joblib.dump(xgb_model, "Baseline_XGBoost_Model.pkl")

