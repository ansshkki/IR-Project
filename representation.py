from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_documents(documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    return vectors, vectorizer
