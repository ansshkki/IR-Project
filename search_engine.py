import nltk
from sklearn.metrics.pairwise import cosine_similarity
from query_processing import process_query
from retrieval import rank_documents
from indexing_storage import build_or_load_index, load_data
from query_refinement import expand_query
from document_clustering import cluster_documents,find_relevant_doc_vector_by_clusters
from topic_detection import detect_topics, print_top_words

# wikir/en1k/training

nltk.download('stopwords')
nltk.download('punkt')

class SearchEngine:
    def __init__(self):
        pass

    def google_search_engine(self, dataset_name: str, query: str, num_docs: int = 10000, top_n: int = 2):
        vectorizer, document_vectors,documents,clusters ,cluster_center = build_or_load_index(dataset_name, num_docs=num_docs)
        
        expanded_query = expand_query(query)
        query_vec = process_query(expanded_query, vectorizer)
        
        # Without cluster
        ranked_indices, similarity_scores = rank_documents(query_vec, document_vectors)
        
        # Print result before optimization 
        print('\n')
        print(f"Top score: {similarity_scores[ranked_indices[0]]} --- for doc index {ranked_indices[0]}")
        print('\n')
        print(f"Best doc data: {documents[ranked_indices[0]]}")
        print('\n')
        
        # --------------------------------
        # Optional: Cluster top documents
        relevant_docs_vectors = find_relevant_doc_vector_by_clusters(query_vec,document_vectors,cluster_center,clusters)
        # Search only in relevant clusters
        ranked_indices_after_cluster, similarity_scores_after_cluster = rank_documents(query_vec, relevant_docs_vectors)
        # Print result after cluster optimization 
        print('\n')
        print(f"Top score cluster: {similarity_scores_after_cluster[ranked_indices_after_cluster[0]]} --- for doc index {ranked_indices_after_cluster[0]}")
        print('\n')
        print(f"Best doc data cluster: {documents[ranked_indices_after_cluster[0]]}")
        print('\n')
        # --------------------------------

        top_documents = [documents[index] for index in ranked_indices[:top_n]]
        results = [(top_documents[i], similarity_scores[i]) for i in range(top_n)]
        
        # Optional: Topic Detection on top documents
        lda_model, lda_vectorizer = detect_topics(top_documents)
        top_words = print_top_words(lda_model, lda_vectorizer.get_feature_names_out(), 20)
        print(f" top words Topic Detection{top_words}")
        
        return {
            "results": [{"document": doc, "score": score} for doc, score in results],
             "topics": top_words
        }


