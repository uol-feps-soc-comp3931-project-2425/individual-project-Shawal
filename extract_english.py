import pandas as pd
from langdetect import detect
import swifter
from tqdm import tqdm

#load the dataset
df = pd.read_csv('eCommerce.csv')

#define the language detection function
def detect_language(text):
    try:
        #Check for empty or NaN values
        if pd.isna(text) or text.strip() == "":  
            return "N/A"
        return detect(text)
    except:
        return "unknown"

#apply the language detection
df['Language'] = df['Customer Remarks'].swifter.apply(detect_language)

#Save the updated dataset with the new 'Language' column
df.to_csv('Language_Labels.csv', index=False)

#filter only English reviews
df_english = df[df['Language'] == 'en']

#save the filtered data to a new CSV
df_english.to_csv('English_eCommerce.csv', index=False)
