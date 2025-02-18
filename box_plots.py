import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load your dataset
df = pd.read_csv('eCommerce.csv')

# #DEBUGGING
# print(df.groupby('category')['CSAT Score'].describe())
# print(df.groupby('category')['CSAT Score'].var())

#create a figure and specify the size
plt.figure(figsize=(12,8))

#create the box plot
sns.boxplot(x='category', y='CSAT Score', data=df, palette='Set2')

#customise the plot
plt.title('CSAT Scores by Category', fontweight='bold')
plt.xlabel('Category', fontstyle='italic')
plt.ylabel('CSAT Score', fontstyle='italic')

#rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

#adjust layout and display
plt.tight_layout()
plt.show()

#violin plot 

print(df['CSAT Score'].describe())
print(df['CSAT Score'].unique())  # Check all unique values
print(df[df['CSAT Score'] > 5])   # Find all problematic rows

#create a figure and specify the size
plt.figure(figsize=(12,8))

#create the violin plot
sns.violinplot(x='category', y='CSAT Score', data=df, palette='Set2')

#customise the plot
plt.title('CSAT Scores by Category', fontweight='bold')
plt.xlabel('Category', fontstyle='italic')
plt.ylabel('CSAT Score', fontstyle='italic')

#rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

plt.show()

