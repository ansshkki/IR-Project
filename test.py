from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF (Term Frequency-Inverse Document Frequency)
# TF-IDF is an improvement over the Bag of Words model. It not only considers the frequency of words in a document but also how common or rare the words are across all documents.

# Steps:
# Term Frequency (TF): Calculate the frequency of each term in each document.
# Inverse Document Frequency (IDF): Calculate the importance of each term in the corpus (logarithmically scaled inverse fraction of the documents that contain the term).
# TF-IDF Score: Multiply TF by IDF for each term in each document.


# Sample processed text documents
processed_texts = [
    "the you",
    "the",
]

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the data and transform the text documents into feature vectors
document_vectors = vectorizer.fit_transform(processed_texts)

# Inspect the resulting feature vectors
print(document_vectors.toarray())
print(vectorizer.get_feature_names_out())


# Word Embeddings (e.g., Word2Vec, GloVe)
# Word embeddings map words to vectors of real numbers in a continuous vector space, capturing semantic meaning.

# Steps:
# Train Embeddings: Use a large corpus to learn the vector representations of words.
# Use Pre-trained Embeddings: Use existing embeddings trained on large datasets (e.g., Word2Vec, GloVe).
# Example:
# Using pre-trained embeddings, you can represent words in a continuous vector space where similar words have similar vectors.


# Explanation of Example
# Vocabulary Building:

# The vectorizer scans all documents and builds a vocabulary of unique words.
# In this case, the vocabulary might be: ["brown", "cat", "fox", "hat", "in", "quick", "the"].
# Vector Creation:

# Each document is converted into a vector based on the vocabulary.
# Document 1 ("the cat in the hat") becomes [0, 1, 0, 1, 1, 0, 2].
# Document 2 ("the quick brown fox") becomes [1, 0, 1, 0, 0, 1, 1].

from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "the cat in the hat",
    "the quick brown fox"
]

vectorizer = CountVectorizer()
bow_vectors = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
print(bow_vectors.toarray())