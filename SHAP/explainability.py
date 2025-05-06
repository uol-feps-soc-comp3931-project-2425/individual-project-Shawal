import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import joblib
import os

#blue-to-pink gradient
blue_pink = LinearSegmentedColormap.from_list(
    "blue_pink", ["#add8e6", "#d291bc", "#f47ac1"], N=256
)
shap.plots.colors.red_blue = blue_pink

#correct feature names 
pretty_names = {
    "order_date_time_hour": "Order Time (Hour)",
    "order_date_time_dayofweek_missing": "Order Day Missing",
    "order_date_time_hour_missing": "Order Time Missing",
    "Survey_response_Date_dayofweek": "Survey Day",
    "issue_responded_hour": "Response Sent Hour",
    "issue_responded_dayofweek": "Response Sent Day",
    "Issue_reported_at_hour": "Issue Reported Hour",

    "Agent_Shift_Morning": "Agent: Morning Shift",
    "Agent_Shift_Evening": "Agent: Evening Shift",
    "Agent_Shift_Split": "Agent: Split Shift",

    "channel_name_Inbound": "Channel: Inbound",
    "channel_name_Outcall": "Channel: Outbound",

    "Item_price": "Item Price",
    "Item_price_missing": "Item Price Missing",

    "Tenure_Bucket_On_Job_Training": "Tenure: On Job Training",
    "Tenure_Bucket_31-60": "Tenure: 31–60 Days",
    "Tenure_Bucket_61-90": "Tenure: 61–90 Days",
    "Tenure_Bucket_>90": "Tenure: Over 90 Days",

    "response_time": "Response Time (mins)",
    "response_time_missing": "Response Time Missing",
    "Sentiment_Score": "Sentiment Score",
    "Confidence": "Sentiment Confidence",
    "category_encoded": "Category (Encoded)",
    "sub_category_encoded": "Subcategory (Encoded)"
}

#grouped features by theme
feature_groups = {
    "Time-Based Features": [
        "order_date_time_hour", "order_date_time_dayofweek_missing",
        "order_date_time_hour_missing", "Survey_response_Date_dayofweek",
        "issue_responded_hour", "issue_responded_dayofweek", "Issue_reported_at_hour"
    ],
    "Agent Features": ["Agent_Shift_Morning", "Agent_Shift_Evening", "Agent_Shift_Split"],
    "Channel Features": ["channel_name_Inbound", "channel_name_Outcall"],
    "Product & Price Features": ["Item_price", "Item_price_missing"],
    "Tenure Features": [
        "Tenure_Bucket_On_Job_Training", "Tenure_Bucket_31-60",
        "Tenure_Bucket_61-90", "Tenure_Bucket_>90"
    ],
    "Response Time Features": ["response_time", "response_time_missing"],
    "Sentiment Features": ["Sentiment_Score", "Confidence"],
    "Clustering Features": ["category_encoded", "sub_category_encoded"]
}

#generalized SHAP visualiser
def plot_shap_group(features, shap_values, X, title, model_name):
    features_in_X = [f for f in features if f in X.columns]
    if not features_in_X:
        #skip if none of the features are present in this dataset
        return  
    
    subset = X[features_in_X]
    shap_subset = shap_values[:, [X.columns.get_loc(f) for f in features_in_X]]
    display_names = [pretty_names.get(f, f) for f in features_in_X]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_subset, subset, show=False, feature_names=display_names, plot_type="dot")

    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=11, width=2)
    cbar.set_ylabel("Feature Value", fontsize=12, labelpad=10)
    cbar.yaxis.set_label_position("right")

    plt.title(f"{title} – {model_name}", fontsize=14)
    plt.xlabel("SHAP value (impact on model output)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"SHAP_{model_name.replace(' ', '_')}_{title.replace(' ', '_')}.png", dpi=300)
    plt.close()

#run SHAP on any model
def explain_model(model_name):
    print(f"Explaining {model_name}...")
    dataset_path = f"../model_datasets/{model_name}_Model_Dataset.csv"
    model_path = f"../training/{model_name}_XGBoost_Model.pkl"

    #load data & model
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["CSAT_Score"])
    model = joblib.load(model_path)

    #SHAP tree explainer 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    for title, features in feature_groups.items():
        plot_shap_group(features, shap_values, X, title, model_name)

#run for all models
model_names = ["Baseline", "Sentiment", "Clustering", "Combined"]
for name in model_names:
    explain_model(name)
