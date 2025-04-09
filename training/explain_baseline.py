import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import joblib

#load model and data
df = pd.read_csv("../model_datasets/Baseline_Model_Dataset.csv")
X = df.drop(columns=["CSAT_Score"])
model = joblib.load("Baseline_XGBoost_Model.pkl")

#custom blue-to-pink gradient
blue_pink = LinearSegmentedColormap.from_list(
    "blue_pink", ["#add8e6", "#d291bc", "#f47ac1"], N=256
)

#set it as SHAP's color gradient
shap.plots.colors.red_blue = blue_pink

#create SHAP explainer and get values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)  
plt.close()

#define pretty feature names for plotting
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
    "response_time_missing": "Response Time Missing"
}

def plot_shap_group(features, shap_values, X, title):
    subset = X[features]
    shap_subset = shap_values[:, [X.columns.get_loc(f) for f in features]]
    
    #get the display names in order
    display_names = [pretty_names.get(f, f) for f in features]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_subset, subset, show=False, feature_names=display_names, plot_type="dot")

    #thicken the color bar 
    cbar = plt.gcf().axes[-1]  
    cbar.tick_params(labelsize=11, width=2)  
    cbar.set_ylabel("Feature Value", fontsize=12, labelpad=10)
    cbar.yaxis.set_label_position("right") 


    plt.title(title, fontsize=14)
    plt.xlabel("SHAP value (impact on model output)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

time_features = [
    "order_date_time_hour", "order_date_time_dayofweek_missing",
    "order_date_time_hour_missing", "Survey_response_Date_dayofweek",
    "issue_responded_hour", "issue_responded_dayofweek", "Issue_reported_at_hour"
]

agent_features = ["Agent_Shift_Morning", "Agent_Shift_Evening", "Agent_Shift_Split"]
channel_features = ["channel_name_Inbound", "channel_name_Outcall"]
product_features = ["Item_price", "Item_price_missing"]
tenure_features = ["Tenure_Bucket_On_Job_Training", "Tenure_Bucket_31-60", "Tenure_Bucket_61-90", "Tenure_Bucket_>90"]
response_features = ["response_time", "response_time_missing"]

plot_shap_group(time_features, shap_values, X, "Time-Based Features")
plot_shap_group(agent_features, shap_values, X, "Agent Features")
plot_shap_group(channel_features, shap_values, X, "Channel Features")
plot_shap_group(product_features, shap_values, X, "Product & Price Features")
plot_shap_group(tenure_features, shap_values, X, "Tenure Features")
plot_shap_group(response_features, shap_values, X, "Response Time Features")
