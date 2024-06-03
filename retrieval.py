from sklearn.metrics.pairwise import cosine_similarity

from utils import time_it


@time_it
def rank_documents(query_vec, document_vectors):
    similarity_scores = cosine_similarity(query_vec, document_vectors).flatten()
    ranked_indices = similarity_scores.argsort()[::-1]
    return ranked_indices, similarity_scores

# cosine_similarity(query_vec, document_vectors): This function calculates the cosine similarity between the query_vec and each vector in document_vectors. It returns a 2D array where each entry represents the similarity score between the query_vec and a document vector.
# .flatten(): This method converts the 2D array of similarity scores into a 1D array for easier handling. Each element in this array corresponds to the similarity score between the query_vec and a document vector.

# similarity_scores will be an array where each element represents the similarity score between the query and a document.
# similarity_scores.argsort() returns the indices that would sort the similarity_scores array in ascending order.