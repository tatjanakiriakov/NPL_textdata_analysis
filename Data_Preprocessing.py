import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

file_path = '/Users/tatjanakiriakov/Desktop/Data Analysis Project/dataset/complaints_processed.csv'

df = pd.read_csv(file_path)

print(df.head())

nltk.download('punkt')
nltk.download('stopwords')

print("Initial DataFrame:")
print(df.head())

# Drop the unnamed index column
df_cleaned = df.drop(columns=['Unnamed: 0'])

print("\nDataFrame after dropping 'Unnamed: 0' column:")
print(df_cleaned.head())

def preprocess_text(text):
    if isinstance(text, float):  # Check if the text is NaN (float)
        return ""
    
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation from tokens
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Apply preprocessing to the narrative column
df_cleaned['cleaned_narrative'] = df_cleaned['narrative'].apply(preprocess_text)

print("\nDataFrame with cleaned narrative:")
print(df_cleaned[['product', 'cleaned_narrative']].head())

cleaned_file_path = '/Users/tatjanakiriakov/Documents/Uni/Data Analysis/dataset/complaints_processed_cleaned.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)


