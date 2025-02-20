import pandas as pd
from langcodes import Language

#load the dataset
df = pd.read_csv('../datasets/Language_Labels.csv')

#ensure "N/A" values are counted 
df['Language'] = df['Language'].fillna("N/A")

#define a function to convert language codes to full names
def get_full_language_name(lang_code):
    if lang_code == "N/A":
        return "Not Available"
    try:
        return Language.make(language=lang_code).display_name()
    except:
        return "Unknown"

#apply the function to convert language codes to full names
df['Language'] = df['Language'].apply(get_full_language_name)

#count the occurrences of each detected language, including "N/A"
language_counts = df['Language'].value_counts()

#print the count of reviews for each detected language
for language, count in language_counts.items():
    print(f"Language: {language}, Reviews: {count}")
