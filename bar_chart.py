import pandas as pd
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv('eCommerce.csv')

#group the data by 'category' and calculate the average CSAT score
avg_csat = df.groupby('category')['CSAT Score'].mean()

#create the bar chart
plt.figure(figsize=(10, 6))
avg_csat.plot(kind='bar', color='pink')

#customise the chart
plt.xlabel('Category', fontstyle='italic')
plt.ylabel('Average CSAT Score', fontstyle='italic')
plt.title('Average CSAT Score by Category', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()     

#display the plot
plt.show()
