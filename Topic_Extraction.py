from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation


file_path = '/Users/tatjanakiriakov/Desktop/Data Analysis Project/dataset/complaints_processed_cleaned.csv'
df = pd.read_csv(file_path)
# Fill missing values with empty string
df['cleaned_narrative'] = df['cleaned_narrative'].fillna('')
# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=30)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_narrative'])

# Create and fit LSA model
lsa_model = TruncatedSVD(n_components=5, random_state=20)
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
# Display top words associated with each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
print("Top words associated with each topic using Latent Semantic Analysis:")
for topic_idx, topic in enumerate(lsa_model.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx+1}: {' '.join(top_words)}")


# Create LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=20)
lda_matrix = lda_model.fit_transform(tfidf_matrix)
# Display top words associated with each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
print("Top words associated with each topic (LDA):")
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx+1}: {' '.join(top_words)}")
