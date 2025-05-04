import pandas as pd
import matplotlib.pyplot as plt

#define the table data
data = [
    ["Unique id", "object", "Unique identifier for the interaction."],
    ["channel_name", "object", "Customer support channel used"],
    ["category", "object", "High-level category of the issue."],
    ["Sub-category", "object", "Detailed subcategory of the issue."],
    ["Customer Remarks", "object", "Textual feedback from the customer."],
    ["Order_id", "object", "Unique identifier for the order."],
    ["order_date_time", "object", "Timestamp when the order was placed."],
    ["Issue_reported at", "object", "Timestamp when customer reported issue."],
    ["issue_responded", "object", "Timestamp when agent responded to issue."],
    ["Survey_response_Date", "object", "Date the CSAT survey was submitted."],
    ["Customer_City", "object", "City where the customer is located."],
    ["Product_category", "object", "Category of the product."],
    ["Item_price", "float64", "Price of the item."],
    ["connected_handling_time", "float64", "Time taken to handle the issue"],
    ["Agent_name", "object", "Name of the customer service agent."],
    ["Supervisor", "object", "Name of the supervisor."],
    ["Manager", "object", "Name of the manager."],
    ["Tenure Bucket", "object", "Grouped agent tenure range."],
    ["Agent Shift", "object", "The work shift of the agent."],
    ["CSAT Score", "int64", "Customer satisfaction score from 1 to 5."]
]

#create DataFrame
df = pd.DataFrame(data, columns=["Column Name", "Data Type", "Definition"])

#plot table
fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))
ax.axis('off')
cell_colours = [['#F5D5E1'] * 3 for _ in range(len(df))]
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')
for col_index in range(len(df.columns)):
    cell = table[0, col_index]
    cell.set_facecolor('#E89EB8')
    cell.set_text_props(weight='bold', color='black')
table.auto_set_column_width(col=list(range(len(df.columns))))

plt.tight_layout()
plt.show()
