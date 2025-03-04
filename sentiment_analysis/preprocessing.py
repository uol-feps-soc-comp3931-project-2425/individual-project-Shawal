import pandas as pd
import re
import emoji
from transformers import AutoTokenizer

#load your dataset
df = pd.read_csv('../datasets/eCommerce.csv')

#replace NaN values 
df['Customer Remarks'] = df['Customer Remarks'].fillna("[NO_REVIEW]") 
#replace empty strings
df['Customer Remarks'] = df['Customer Remarks'].replace("", "[NO_REVIEW]") 

#load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#function to clean text
def clean_text(text):
     #ensure the input is a string
    if isinstance(text, str): 
        #convert emojis to text
        text = emoji.demojize(text)  
         #remove special characters (keep punctuation)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text) 
         #remove extra spaces
        text = re.sub(r"\s+", " ", text).strip() 
    return text

#apply text cleaning
df['Customer Remarks'] = df['Customer Remarks'].apply(clean_text)

#calculate word count for each review
df['word_count'] = df['Customer Remarks'].apply(lambda x: len(str(x).split()))

#find 95th percentile word count
percentile_95 = df['word_count'].quantile(0.95) 

#convert word count to token count
max_length = min(512, int(percentile_95 * 1.5))

#tokenization function
def tokenize_text(text):
    return tokenizer(
        text, 
        #pads short sequences
        padding="max_length",
        #truncate long sentences to max_length  
        truncation=True,  
        max_length=max_length,           
        return_tensors="pt"   
    )

#tokenize the Customer Remarks column
df['tokens'] = df['Customer Remarks'].apply(lambda x: tokenize_text(x)['input_ids'].squeeze().tolist())

#save the processed dataset
df.to_csv('../datasets/tokenized_eCommerce.csv', index=False)

