from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# Topic Detection
# Purpose: Identifying topics within documents to enhance the retrieval system by understanding document content better.

def detect_topics(documents, num_topics=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    return lda, vectorizer


# Print the top words for each topic for debugging
def print_top_words(model, feature_names, n_top_words):
    message = ''
    for topic_idx, topic in enumerate(model.components_):
        message += f"Topic #{topic_idx}:"
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return message
        