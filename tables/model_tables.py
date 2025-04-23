import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#define model performance data
data = {
    "Model": ["Baseline", "Sentiment", "Clustering", "Combined"],
    "Train/Test RMSE": [1.3500, 1.2293, 1.3312, 1.2133],
    "Train/Test R²": [0.0353, 0.2001, 0.0620, 0.2208],
    "5-Fold Avg RMSE": [1.3531, 1.2388, 1.3356, 1.2250],
    "5-Fold Avg R²": [0.0370, 0.1928, 0.0618, 0.2107],
    "Best Tuned RMSE": [1.3521, 1.2384, 1.3339, 1.2229]
}

#convert to DataFrame
df = pd.DataFrame(data)

#create a pink-themed table
plt.figure(figsize=(10, 2.5))
sns.set(style="whitegrid")
plt.axis('off')

#add the table
table = plt.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#E89EB8'] * len(df.columns),
    cellColours=[['#F5D5E1']*len(df.columns) for _ in range(len(df))]
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

#title
plt.title("Model Data", fontsize=16, color="black", pad=5)
plt.tight_layout()
plt.savefig("model_table.png", dpi=300)
plt.show()
