import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
df = pd.read_csv('../eCommerce.csv')

#convert 'Issue_reported at' to datetime
df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], errors='coerce')

#extract just the date for daily granularity
df['Date'] = df['Issue_reported at'].dt.date

#group by date and category to get the average CSAT score
grouped_df = df.groupby(['Date', 'category'])['CSAT Score'].mean().reset_index()

#create a relplot with one facet per category
g = sns.relplot(
    data=grouped_df,
    x='Date',
    y='CSAT Score',
    col='category',
    kind='line',
    col_wrap=4,
    height=3,
    aspect=1.5
)

#set axis labels for each facet
g.set_axis_labels("Date (YYYY-MM-DD)", "Average CSAT Score")

g.set_titles("Category = {col_name}", fontweight='bold')

g.fig.suptitle("Average CSAT Score by Category Over Time", y=1.02, fontweight='bold')

#rotate x-axis labels for each subplot
for ax in g.axes.flatten():
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()