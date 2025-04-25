import matplotlib.pyplot as plt
import numpy as np

#model names
models = ["Baseline", "Sentiment", "Clustering", "Combined"]

#RMSE values
train_test_rmse = [1.3500, 1.2293, 1.3312, 1.2133]
cv_rmse = [1.3531, 1.2388, 1.3356, 1.2250]
tuned_rmse = [1.3521, 1.2384, 1.3339, 1.2229]

#R² values
train_test_r2 = [0.0353, 0.2001, 0.0620, 0.2208]
cv_r2 = [0.0370, 0.1928, 0.0618, 0.2107]
tuned_r2 = [0.0354, 0.2012, 0.0657, 0.2208]

#X positions and width
x = np.arange(len(models))
width = 0.25

#plot 1: RMSE
plt.figure(figsize=(10, 6))
plt.bar(x - width, train_test_rmse, width, label='Train/Test RMSE', color='#f4a7b9')
plt.bar(x, cv_rmse, width, label='5-Fold RMSE', color='#e68aa2')
plt.bar(x + width, tuned_rmse, width, label='Best Tuned RMSE', color='#d2648c')

plt.xlabel("Models", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE Comparison Across Evaluation Stages", fontsize=14, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.savefig("rmse_comparison.png", dpi=300)
plt.show()

#plot 2: R²
plt.figure(figsize=(10, 6))
plt.bar(x - width, train_test_r2, width, label='Train/Test R²', color='#cdb4db')
plt.bar(x, cv_r2, width, label='5-Fold R²', color='#b197c2')
plt.bar(x + width, tuned_r2, width, label='Best Tuned R²', color='#9d6aae')

plt.xlabel("Models", fontsize=12)
plt.ylabel("R² Score", fontsize=12)
plt.title("R² Comparison Across Evaluation Stages", fontsize=14, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.savefig("r2_comparison.png", dpi=300)
plt.show()
