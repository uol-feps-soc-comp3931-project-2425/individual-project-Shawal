import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langcodes import Language

#load the dataset
df = pd.read_csv('datasets/Language_Labels.csv')  # Ensure correct path

#function to convert language codes to full names
def get_full_language_name(lang_code):
    if pd.isna(lang_code) or lang_code.strip() == "":
        return "Not Available"
    try:
        return Language.make(language=lang_code).display_name()
    except:
        return "Unknown"

#convert language codes to full names
df['Language'] = df['Language'].apply(get_full_language_name)

#plot the violin plot for CSAT score distribution per language
plt.figure(figsize=(14, 6))

sns.violinplot(x='Language', y='CSAT Score', data=df, palette=sns.color_palette("pastel"))

plt.xlabel('Language')
plt.ylabel('CSAT Score')
plt.title('CSAT Score Distribution by Language')

plt.xticks(rotation=90, ha="right")

plt.tight_layout()
plt.show()
