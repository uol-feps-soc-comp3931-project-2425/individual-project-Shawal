import pandas as pd
from sklearn.preprocessing import LabelEncoder

#load the dataset
df = pd.read_csv("../datasets/Final.csv")  

#create a copy for the Baseline Model
df_baseline = df.copy()

#drop segmentation-related features
segmentation_features = ["category", "Sub-category"]
df_baseline.drop(columns=segmentation_features, inplace=True, errors="ignore")

datetime_cols = ["order_date_time", "Issue_reported at", "issue_responded", "Survey_response_Date"]

for dtcol in datetime_cols:
    if dtcol in df_baseline.columns:
        #convert to datetime, coerce errors to NaT (Not a Time)
        df_baseline[dtcol] = pd.to_datetime(df_baseline[dtcol], errors="coerce")
        
        #create new columns for hour & day of week
        df_baseline[f"{dtcol}_hour"] = df_baseline[dtcol].dt.hour
        df_baseline[f"{dtcol}_dayofweek"] = df_baseline[dtcol].dt.dayofweek
        
        #drop the original datetime column
        df_baseline.drop(columns=[dtcol], inplace=True)

#remove sentiment-related features
sentiment_features = ["Sentiment", "Confidence", "tokens", "word_count", "Customer Remarks"]
df_baseline.drop(columns=[col for col in sentiment_features if col in df_baseline.columns],
                 inplace=True, errors="ignore")

#convert categorical variables into numerical values
categorical_cols = df_baseline.select_dtypes(include=["object", "category"]).columns.tolist()

#drop unecessary ID columns 
id_like_cols = ["Unique id", "Order_id", "Agent_name", "Supervisor", "Manager"]
categorical_cols = [col for col in categorical_cols if col not in id_like_cols]
#drop ID-like columns from final dataset
df_baseline.drop(columns=id_like_cols, inplace=True, errors="ignore")

#one-hot encode the remaining categorical columns
df_baseline = pd.get_dummies(df_baseline, columns=categorical_cols, drop_first=True)

#for each numeric column, create an indicator and fill missing with 0
numeric_cols = df_baseline.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    #create a new column marking whether the original value was missing
    df_baseline[f"{col}_missing"] = df_baseline[col].isna().astype(int)
    #fill the missing values with 0
    df_baseline[col].fillna(0, inplace=True)

#convert boolean columns (True/False) to integers (1/0)
df_baseline = df_baseline.astype({col: int for col in df_baseline.select_dtypes(bool).columns})

#clean column names: remove spaces and replace with underscores
df_baseline.columns = df_baseline.columns.str.replace(" ", "_")

df_baseline.to_csv("Baseline_Model_Dataset.csv", index=False)
