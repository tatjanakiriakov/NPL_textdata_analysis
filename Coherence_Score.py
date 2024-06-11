import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Load data
file_path = '/Users/tatjanakiriakov/Desktop/Data Analysis Project/dataset/complaints_processed_cleaned.csv'
df = pd.read_csv(file_path)

# Fill missing values with empty string
df['cleaned_narrative'] = df['cleaned_narrative'].fillna('')

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=30)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_narrative'])

# Tokenize documents for coherence score calculation
tokenized_docs = [doc.split() for doc in df['cleaned_narrative']]

# Create dictionary and corpus required for CoherenceModel
dictionary = Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

# Create and fit LSA model
lsa_model = TruncatedSVD(n_components=5, random_state=20)
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

# Display top words associated with each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
print("Top words associated with each topic using Latent Semantic Analysis:")
lsa_topics = []
for topic_idx, topic in enumerate(lsa_model.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    lsa_topics.append(top_words)
    print(f"Topic {topic_idx+1}: {' '.join(top_words)}")

# Calculate coherence score for LSA
lsa_coherence_model = CoherenceModel(topics=lsa_topics, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
lsa_coherence_score = lsa_coherence_model.get_coherence()
print(f"LSA Coherence Score: {lsa_coherence_score}")

# Create LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=20)
lda_matrix = lda_model.fit_transform(tfidf_matrix)

# Display top words associated with each topic
print("Top words associated with each topic (LDA):")
lda_topics = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    lda_topics.append(top_words)
    print(f"Topic {topic_idx+1}: {' '.join(top_words)}")

# Convert LDA topics to format expected by CoherenceModel
lda_topics_formatted = [[feature_names[word_idx] for word_idx in topic.argsort()[:-11:-1]] for topic in lda_model.components_]

# Calculate coherence score for LDA
lda_coherence_model = CoherenceModel(topics=lda_topics_formatted, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
lda_coherence_score = lda_coherence_model.get_coherence()
print(f"LDA Coherence Score: {lda_coherence_score}")



