import nltk
from sklearn.metrics.pairwise import cosine_similarity
from query_processing import process_query
from retrieval import rank_documents
from indexing_storage import build_or_load_index, load_data
from query_refinement import expand_query
from document_clustering import cluster_documents,print_clusters
from topic_detection import detect_topics, print_top_words

# wikir/en1k/training

nltk.download('stopwords')
nltk.download('punkt')

class SearchEngine:
    def __init__(self):
        pass

    def google_search_engine(self, dataset_name: str, query: str, num_docs: int = 10000, top_n: int = 2):
        vectorizer, document_vectors = build_or_load_index(dataset_name, num_docs=num_docs)
        
        expanded_query = expand_query(query)
        query_vec = process_query(expanded_query, vectorizer)
        
        ranked_indices, similarity_scores = rank_documents(query_vec, document_vectors)

        documents = load_data(dataset_name, num_docs=num_docs)

        if len(documents) == 0:
            return {"error": "No documents found in the dataset."}

        top_documents = [documents[index] for index in ranked_indices[:top_n]]
        
        results = [(top_documents[i], similarity_scores[i]) for i in range(top_n)]
        
        # Optional: Cluster top documents
        cluster_labels = cluster_documents(document_vectors)
        cluster_results = print_clusters(top_documents,cluster_labels)

        # Optional: Topic Detection on top documents
        lda_model, lda_vectorizer = detect_topics(top_documents)
        top_words = print_top_words(lda_model, lda_vectorizer.get_feature_names_out(), 20)
        
        return {
            "results": [{"document": doc, "score": score} for doc, score in results],
            "clusters": cluster_results,
            "topics": top_words
        }
