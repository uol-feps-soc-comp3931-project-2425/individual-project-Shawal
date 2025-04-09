import pandas as pd

#load baseline features
baseline_df = pd.read_csv("../model_datasets/Baseline_Model_Dataset.csv")

#load original full dataset (with category/sub-category)
raw_df = pd.read_csv("../datasets/eCommerce.csv")  

#merge by row order
baseline_df["category"] = raw_df["category"]
baseline_df["sub_category"] = raw_df["Sub-category"]

#encode category and sub-category numerically
baseline_df["category_encoded"] = baseline_df["category"].astype("category").cat.codes
baseline_df["sub_category_encoded"] = baseline_df["sub_category"].astype("category").cat.codes

baseline_df.drop(columns=["category", "sub_category"], inplace=True)

#save final clustering model dataset
baseline_df.to_csv("../model_datasets/Clustering_Model_Dataset.csv", index=False)