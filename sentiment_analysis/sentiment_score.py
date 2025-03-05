import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#load the dataset
df = pd.read_csv('../datasets/tokenized_eCommerce.csv')

#load sentiment model & tokenizer 
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
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
    
    #3-class sentiment mapping
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment_label = sentiment_mapping[predicted_class]
    
    return sentiment_label, confidence

#save the results
df.to_csv('../datasets/sentiment_analysis_results_3class.csv', index=False)
