# Since we're using TF-IDF, the indexing is part of the vectorization process in representation.py.
def build_index(vectorizer, documents):
    X = vectorizer.fit_transform(documents)
    return X
