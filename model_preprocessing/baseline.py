#Use XGBoost Regressor 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

#load the dataset
dataset = pd.read_csv('../datasets/Final.csv')

#identify categorical columns
cat_cols = ['channel_name', 'category', 'Sub-category', 'Customer_City', 'Product_category',
            'Agent_name', 'Supervisor', 'Manager', 'Tenure Bucket', 'Agent Shift']

#lill missing categorical values before encoding
dataset[cat_cols] = dataset[cat_cols].fillna("Unknown")

#apply label encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col].astype(str))  
    label_encoders[col] = le  

#keep missing numerical values as NaN
num_cols = ['Item_price', 'response_time', 'connected_handling_time']
dataset[num_cols] = dataset[num_cols].replace("Unknown", pd.NA)

dataset["CSAT Score"] = pd.to_numeric(dataset["CSAT Score"], errors='coerce')

#convert date columns
date_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
for col in date_cols:
    dataset[col] = pd.to_datetime(dataset[col], errors='coerce')

#extract features
for col in date_cols:
    dataset[col + '_year'] = dataset[col].dt.year
    dataset[col + '_month'] = dataset[col].dt.month
    dataset[col + '_day'] = dataset[col].dt.day
    dataset[col + '_hour'] = dataset[col].dt.hour

#drop original datetime columns
dataset.drop(columns=date_cols, inplace=True)

# Define features manually
selected_features = [
    'channel_name', 'Customer_City', 'Product_category',
    'Agent_name', 'Supervisor', 'Manager', 'Tenure Bucket', 'Agent Shift',
    'Item_price', 'response_time', 'connected_handling_time',
    'order_date_time_year', 'order_date_time_month', 'order_date_time_day', 'order_date_time_hour',
    'Issue_reported at_year', 'Issue_reported at_month', 'Issue_reported at_day', 'Issue_reported at_hour',
    'issue_responded_year', 'issue_responded_month', 'issue_responded_day', 'issue_responded_hour',
    'Survey_response_Date_year', 'Survey_response_Date_month', 'Survey_response_Date_day', 'Survey_response_Date_hour'
]

#define features (X) and target variable (y)
X = dataset[selected_features]
y = dataset["CSAT Score"]

#split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the processed dataset to a CSV file
output_file_path = "Processed_Baseline.csv"
dataset.to_csv(output_file_path, index=False)




