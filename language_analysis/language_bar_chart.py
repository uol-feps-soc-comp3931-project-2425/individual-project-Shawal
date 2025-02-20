import matplotlib.pyplot as plt
import pandas as pd
from langcodes import Language

#load the dataset
df = pd.read_csv('../datasets/Language_Labels.csv')

#function to convert language codes to full names
def get_full_language_name(lang_code):
    if pd.isna(lang_code) or lang_code.strip() == "":
        return "Not Available"
    try:
        return Language.make(language=lang_code).display_name()
    except:
        return "Unknown"

#apply conversion to full language names
df['Language'] = df['Language'].apply(get_full_language_name)

#calculate the average CSAT score per language
csat_by_language = df.groupby('Language')['CSAT Score'].mean().sort_values()

#plot the average CSAT score per language
plt.figure(figsize=(12, 6))

csat_by_language.plot(kind='bar', color='thistle')

plt.xlabel('Language')
plt.ylabel('Average CSAT Score')
plt.title('Average CSAT Score by Language')

plt.xticks(rotation=90, ha="right")

plt.tight_layout()
plt.show()
