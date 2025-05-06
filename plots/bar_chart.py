import matplotlib.pyplot as plt
import numpy as np

# Model names and values
models = ["Baseline", "Sentiment", "Clustering", "Combined"]

# RMSE values
train_test_rmse = [1.3500, 1.2293, 1.3312, 1.2133]
cv_rmse = [1.3531, 1.2388, 1.3356, 1.2250]
tuned_rmse = [1.3521, 1.2384, 1.3339, 1.2229]

# R² values
train_test_r2 = [0.0353, 0.2001, 0.0620, 0.2208]
cv_r2 = [0.0370, 0.1928, 0.0618, 0.2107]
tuned_r2 = [0.0354, 0.2012, 0.0657, 0.2208]

# Sort based on best tuned R² descending
sorted_indices = np.argsort(tuned_r2)[::-1]
models_sorted = [models[i] for i in sorted_indices]
train_test_rmse_sorted = [train_test_rmse[i] for i in sorted_indices]
cv_rmse_sorted = [cv_rmse[i] for i in sorted_indices]
tuned_rmse_sorted = [tuned_rmse[i] for i in sorted_indices]
train_test_r2_sorted = [train_test_r2[i] for i in sorted_indices]
cv_r2_sorted = [cv_r2[i] for i in sorted_indices]
tuned_r2_sorted = [tuned_r2[i] for i in sorted_indices]

# Bar chart settings
x = np.arange(len(models_sorted))
width = 0.25

# Plot 1: RMSE
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width, train_test_rmse_sorted, width, label='Train/Test RMSE', color='#f4a7b9')
bars2 = plt.bar(x, cv_rmse_sorted, width, label='5-Fold RMSE', color='#e68aa2')
bars3 = plt.bar(x + width, tuned_rmse_sorted, width, label='Best Tuned RMSE', color='#d2648c')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f'{yval:.4f}', ha='center', fontsize=8)

plt.xlabel("Models", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE Comparison Across Evaluation Stages", fontsize=14, fontweight='bold')
plt.xticks(x, models_sorted)
plt.legend()
plt.tight_layout()
plt.savefig("rmse_comparison_labeled.png", dpi=300)
plt.show()

# Plot 2: R²
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width, train_test_r2_sorted, width, label='Train/Test R²', color='#cdb4db')
bars2 = plt.bar(x, cv_r2_sorted, width, label='5-Fold R²', color='#b197c2')
bars3 = plt.bar(x + width, tuned_r2_sorted, width, label='Best Tuned R²', color='#9d6aae')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f'{yval:.4f}', ha='center', fontsize=8)

plt.xlabel("Models", fontsize=12)
plt.ylabel("R² Score", fontsize=12)
plt.title("R² Comparison Across Evaluation Stages", fontsize=14, fontweight='bold')
plt.xticks(x, models_sorted)
plt.legend()
plt.tight_layout()
plt.savefig("r2_comparison_labeled.png", dpi=300)
plt.show()
