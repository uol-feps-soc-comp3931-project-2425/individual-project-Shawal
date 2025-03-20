import pandas as pd

#load the dataset
df = pd.read_csv('datasets/Sentiment_Analysis_Results.csv')

#convert to datetime format if not already
df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], errors='coerce')
df['issue_responded'] = pd.to_datetime(df['issue_responded'], errors='coerce')

#calculate response time (in minutes)
df['response_time'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60

#ensure no negative response times due to potential data errors
df['response_time'] = df['response_time'].apply(lambda x: x if x >= 0 else None)

#save the updated dataset
df.to_csv('datasets/Final.csv', index=False)

