import pandas as pd
import matplotlib.pyplot as plt

# #CATEGORY

#load the dataset
df = pd.read_csv('eCommerce.csv')

# #group the data by 'category' and calculate the average CSAT score
# avg_csat = df.groupby('category')['CSAT Score'].mean()

# #create the bar chart
# plt.figure(figsize=(10, 6))
# avg_csat.plot(kind='bar', color='pink')

# #customise the chart
# plt.xlabel('Category', fontstyle='italic')
# plt.ylabel('Average CSAT Score', fontstyle='italic')
# plt.title('Average CSAT Score by Category', fontweight='bold')
# plt.xticks(rotation=45)
# plt.tight_layout()     

# #display the plot
# plt.show()

#SUBCATEGORY 

import math

#group the data by 'category' and 'Sub-category' to calculate the average CSAT score
subcat_csat = df.groupby(['category', 'Sub-category'])['CSAT Score'].mean().reset_index()

#get the unique categories
unique_categories = subcat_csat['category'].unique()
n_categories = len(unique_categories)

# Set number of columns and calculate rows needed
n_cols = min(4, n_categories)
n_rows = math.ceil(n_categories / n_cols)

#create a subplot for each category
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 4, n_rows * 4))
axes = axes.flatten()

#loop over each category to create a horizontal bar chart per subplot
for ax, category in zip(axes, unique_categories):
    #filter data for the current category
    data = subcat_csat[subcat_csat['category'] == category]
    
    ax.barh(data['Sub-category'], data['CSAT Score'], color='skyblue', height=0.5)
    
    #set subplot title and labels
    ax.set_title(f'Average CSAT Score by {category}', fontweight='bold')
    ax.set_xlabel('Sub-Category', fontstyle='italic')
    ax.set_ylabel('Average CSAT Score', fontstyle='italic')

#adjust layout to prevent overlapping elements
plt.subplots_adjust(hspace=0.5, wspace=2.0)
plt.show()


