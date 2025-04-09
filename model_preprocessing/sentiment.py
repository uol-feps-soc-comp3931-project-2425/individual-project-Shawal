import pandas as pd

#load baseline features
baseline_df = pd.read_csv("../model_datasets/Baseline_Model_Dataset.csv")

#load sentiment results
sentiment_df = pd.read_csv("../datasets/Sentiment_Analysis_Results.csv")

#check for common columns
#print(baseline_df.columns)
#print(sentiment_df.columns)

#merge by row order 
merged_df = pd.concat(
    [baseline_df.reset_index(drop=True),
     sentiment_df[["Sentiment", "Confidence"]].reset_index(drop=True)],
    axis=1
)

#map sentiment labels to numeric scores
sentiment_map = {
    "Very Negative": 1,
    "Negative": 2,
    "Neutral": 3,
    "Positive": 4,
    "Very Positive": 5
}
merged_df["Sentiment_Score"] = merged_df["Sentiment"].map(sentiment_map)

#remove raw text columns
merged_df = merged_df.drop(columns=["Customer Remarks", "word_count", "tokens"], errors="ignore")

#save your sentiment-enhanced dataset
merged_df.to_csv("../model_datasets/Sentiment_Model_Dataset.csv", index=False)