import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Load dataset
df = pd.read_csv("../datasets/eCommerce.csv")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

# Corrected column definitions based on dataset
column_definitions = {
    "Unique id": "Unique identifier for the interaction.",
    "channel_name": "Customer support channel used",
    "category": "High-level category of the issue.",
    "Sub-category": "Detailed subcategory of the issue.",
    "Customer Remarks": "Textual feedback from the customer.",
    "Order_id": "Unique identifier for the order.",
    "order_date_time": "Timestamp when the order was placed.",
    "Issue_reported at": "Timestamp when customer reported issue.",
    "issue_responded": "Timestamp when agent responded to issue.",
    "Survey_response_Date": "Date the CSAT survey was submitted.",
    "Customer_City": "City where the customer is located.",
    "Product_category": "Category of the product.",
    "Item_price": "Price of the item.",
    "connected_handling_time": "Time taken to handle the issue",
    "Agent_name": "Name of the customer service agent.",
    "Supervisor": "Name of the supervisor.",
    "Manager": "Name of the manager.",
    "Tenure Bucket": "Grouped agent tenure range.",
    "Agent Shift": "The work shift of the agent.",
    "CSAT Score": "Customer satisfaction score from 1 to 5."
}

# Build the schema table
schema_df = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.values,
    "Definition": [column_definitions.get(col, "—") for col in df.columns]
})

# Wrap the definitions to avoid overflow
wrapped_definitions = [
    "\n".join(textwrap.wrap(str(defn), width=40)) for defn in schema_df["Definition"]
]
schema_df["Definition"] = wrapped_definitions

# Plotting the schema as a pink-themed table
plt.figure(figsize=(16, len(schema_df) * 0.8))  # 👈 wider and taller
sns.set(style="whitegrid")
plt.axis('off')

# Build the table
table = plt.table(
    cellText=schema_df.values,
    colLabels=schema_df.columns,
    cellLoc='left',
    loc='center',
    colColours=['#E89EB8'] * schema_df.shape[1],
    cellColours=[['#F5D5E1'] * schema_df.shape[1] for _ in range(schema_df.shape[0])],
    colLoc='left'
)

# Style adjustments
table.auto_set_font_size(False)
table.set_fontsize(13)  # 👈 Increase font size
table.scale(1.2, 1.5)

# Make column headers bold
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Header row
        cell.set_text_props(weight='bold')

# Save as image
plt.tight_layout()
#plt.figure(figsize=(14, len(schema_df) * 0.6))  # wider and taller
plt.savefig("dataset_schema_pink.png", dpi=800)
plt.close()

