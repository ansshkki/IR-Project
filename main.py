import nltk
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Import custom modules
from query_processing import process_query
from retrieval import rank_documents
from indexing_storage import build_or_load_index, load_data
from query_refinement import expand_query
from document_clustering import cluster_documents,print_clusters
from topic_detection import detect_topics, print_top_words

# Dataset name
dataset_name = 'wikir/en1k/training'

# Number of documents to process
num_docs = 100000  # Adjust this value to process a different number of documents

# Build or load the index
vectorizer, document_vectors = build_or_load_index(dataset_name, num_docs=num_docs)

# Example query
query = "during his time in t he held the posts of cabinet secretary chairman of "

# Expand query
expanded_query = expand_query(query)

query_vec = process_query(expanded_query, vectorizer)

# Rank documents
ranked_indices, similarity_scores = rank_documents(query_vec, document_vectors)

# Load the subset of the dataset to map indices to original documents
documents = load_data(dataset_name, num_docs=num_docs)

lda_model, lda_vectorizer = detect_topics(documents)
print_top_words(lda_model, lda_vectorizer.get_feature_names_out(), 10)


# Output the ranked documents
for index in ranked_indices[:2]:  # Output top 10 results
    # print(documents[index], similarity_scores[index])
    print(similarity_scores[index])
    print(documents[index])


# Cluster documents
clusters = cluster_documents(document_vectors)
print_clusters(documents, clusters)  # Print cluster details


# Topic Detection
lda_model, lda_vectorizer = detect_topics(documents)
print_top_words(lda_model, lda_vectorizer.get_feature_names_out(), 10)