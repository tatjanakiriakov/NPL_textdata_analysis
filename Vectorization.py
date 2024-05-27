
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


cleaned_file_path = '/Users/tatjanakiriakov/Documents/Uni/Data Analysis/dataset/complaints_processed_cleaned.csv'
df = pd.read_csv(cleaned_file_path)


#Bag of Words Vectorization
file_path = '/Users/tatjanakiriakov/Desktop/Data Analysis Project/dataset/complaints_processed_cleaned.csv'
df = pd.read_csv(file_path)
df['cleaned_narrative'] = df['cleaned_narrative'].fillna('')
# Create CountVectorizer 
bow_vectorizer = CountVectorizer(max_features=20)  # adjust max_features 
# Fit and transform the text data
bow_matrix = bow_vectorizer.fit_transform(df['cleaned_narrative'])
# Convert the BoW matrix to DataFrame 
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
# Print the Bag of Words DataFrame
print("Bag of Words DataFrame:")
print(bow_df.head())


# TF-DFI Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=20)  #  adjust max_features 
df['cleaned_narrative'] = df['cleaned_narrative'].fillna('')
# Fit data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_narrative'])
# Convert the TF-IDF matrix to a DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# Print the TF-IDF DataFrame
print("TF-IDF DataFrame:")
print(tfidf_df.head())
