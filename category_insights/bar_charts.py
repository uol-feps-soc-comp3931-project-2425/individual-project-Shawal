import pandas as pd
import matplotlib.pyplot as plt

#CATEGORY

#load the dataset
df = pd.read_csv('../eCommerce.csv')

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

#SUBCATEGORY 

import math

selected_categories = ['Order Related', 'Others']
#'Order Related', 'Product Queries', 'Others', 'Cancellation'

#filter the dataset to only include rows from the selected categories
filtered_df = df[df['category'].isin(selected_categories)]

#group the data by 'category' and 'Sub-category' to calculate the average CSAT score
subcat_csat = filtered_df.groupby(['category', 'Sub-category'])['CSAT Score'].mean().reset_index()

#get the unique categories
unique_categories = subcat_csat['category'].unique()

#create a subplot for each selected category
fig, axes = plt.subplots(nrows=len(unique_categories), ncols=1, figsize=(12, len(unique_categories) * 4))

#loop over each category to create a horizontal bar chart per subplot
for ax, category in zip(axes, unique_categories):
    #filter data for the current category
    data = subcat_csat[subcat_csat['category'] == category]
    
    ax.barh(data['Sub-category'], data['CSAT Score'], color='skyblue', height=0.7)
    
    #set subplot title and labels
    ax.set_title(f'Average CSAT Score by {category}', fontweight='bold')
    ax.set_xlabel('Sub-Category', fontstyle='italic')
    ax.set_ylabel('Average CSAT Score', fontstyle='italic')
    
#adjust layout to prevent overlapping elements
plt.tight_layout()
plt.show()

#print the results
print(avg_csat)
print(subcat_csat)

