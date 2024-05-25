from sklearn.metrics.pairwise import cosine_similarity

def rank_documents(query_vec, document_vectors):
    similarity_scores = cosine_similarity(query_vec, document_vectors).flatten()
    ranked_indices = similarity_scores.argsort()[::-1]
    return ranked_indices, similarity_scores
