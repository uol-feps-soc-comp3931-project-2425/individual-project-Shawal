import pandas as pd

#load all datasets
baseline_df = pd.read_csv("../model_datasets/Baseline_Model_Dataset.csv")
sentiment_df = pd.read_csv("../model_datasets/Sentiment_Model_Dataset.csv")
clustering_df = pd.read_csv("../model_datasets/Clustering_Model_Dataset.csv")

#extract just the new columns from sentiment & clustering
sentiment_cols = sentiment_df[["Sentiment_Score", "Confidence"]]
clustering_cols = clustering_df[["category_encoded", "sub_category_encoded"]]

#combine all into one DataFrame
combined_df = pd.concat(
    [baseline_df.reset_index(drop=True),
     sentiment_cols.reset_index(drop=True),
     clustering_cols.reset_index(drop=True)],
    axis=1
)

#save combined model dataset
combined_df.to_csv("../model_datasets/Combined_Model_Dataset.csv", index=False)