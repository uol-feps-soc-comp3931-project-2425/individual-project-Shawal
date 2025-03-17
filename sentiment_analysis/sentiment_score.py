import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 #import tqdm for progress bar
from tqdm import tqdm 

#load the dataset
df = pd.read_csv('../datasets/tokenized_eCommerce.csv')

#load sentiment model & tokenizer 
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentiment(text):
    #convert non-string inputs to string
    if not isinstance(text, str): 
        text = "[NO_REVIEW]"
    #skip processing for missing reviews
    if text == "[NO_REVIEW]":  
        #assign neutral label with 0 confidence
        return "No Review", 0.0  
    
    inputs = tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    with torch.no_grad():  
        outputs = model(**inputs)
    
    logits = outputs.logits
    #get the predicted label
    predicted_class = torch.argmax(logits, dim=1).item()  
    #confidence score
    confidence = torch.softmax(logits, dim=1).max().item()  
    
    #5-class sentiment mapping
    sentiment_mapping = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }
    sentiment_label = sentiment_mapping[predicted_class]
    
    return sentiment_label, confidence

tqdm.pandas(desc="Processing Sentiment Analysis")
df[['Sentiment', 'Confidence']] = df['Customer Remarks'].progress_apply(lambda x: pd.Series(get_sentiment(x)))

#save the results
df.to_csv('../datasets/Sentiment_Analysis_Results.csv', index=False)
