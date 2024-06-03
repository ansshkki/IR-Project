# Documents Clustering
# Purpose: Grouping similar documents together to understand the structure of the data and improve retrieval.

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from utils import time_it


def find_optimal_num_clusters(data, max_clusters=10):
    """
    Find the optimal number of clusters using the Elbow Method.
    
    Parameters:
        data (numpy.ndarray or scipy.sparse matrix): Input data for clustering.
        max_clusters (int): Maximum number of clusters to consider.
    
    Returns:
        int: Optimal number of clusters.
    """
    wcss = []  # Within-cluster sum of squares

    for k in range(1, max_clusters + 1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        wcss.append(km.inertia_)

    # Plot the Elbow Method curve
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    # plt.title('Elbow Method')
    # plt.show()

    # Find the "elbow point"
    deltas = np.diff(wcss, 2)
    optimal_num_clusters = np.argmax(deltas) + 2  # Add 2 due to 0-based indexing and the differentiation step

    return optimal_num_clusters
# Example usage:
# optimal_clusters = find_optimal_num_clusters(document_vectors)


# -----------------------------------------------------------------------------------------

def cluster_documents(document_vectors):
    num_clusters = find_optimal_num_clusters(document_vectors)
    print(f"num_clusters: {num_clusters}")
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(document_vectors)
    clusters = km.labels_
    cluster_center = km.cluster_centers_
    return clusters,cluster_center

# n_clusters=num_clusters specifies the number of clusters (here, 2).
# random_state=42 is a seed value for the random number generator used by the algorithm. Setting this ensures that the results are reproducible. If you run the algorithm again with the same random state and data, you'll get the same clustering results.

# km.fit(document_vectors):
# This method fits the K-means algorithm to your document vectors. Essentially, it runs the clustering algorithm on the data.

# clusters = km.labels_:
# After fitting the model, km.labels_ contains the cluster labels for each document.
# clusters is an array where each element corresponds to a document and its value is the cluster number (0, 1, etc.) to which the document has been assigned.

# -----------------------------------------------------------------------------------------

# Function to find the relevant clusters for a query
@time_it
def find_relevant_doc_vector_by_clusters(query_vec,document_vectors,cluster_centers, clusters):
    similarity_to_clusters = cosine_similarity(query_vec, cluster_centers)
    relevant_cluster_indices = similarity_to_clusters.argsort()[0][::-1]
    relevant_docs_indices = np.where(np.isin(clusters, relevant_cluster_indices))[0]
    relevant_docs_vectors = document_vectors[relevant_docs_indices]
    return relevant_docs_vectors

# -----------------------------------------------------------------------------------------


#  Document clustering is a technique used to group a set of documents into clusters, where documents in the same cluster are more similar to each other than to those in other clusters. This can help in organizing large datasets, improving search results, and discovering patterns or topics within the documents.

# Documents Clustering
# Purpose:
# Document clustering helps in:

# Organizing large datasets: By grouping similar documents together, it becomes easier to manage and browse large collections.
# Improving search results: Search results can be grouped by topic, making it easier for users to find relevant information.
# Topic discovery: Clustering can reveal underlying themes or topics within the dataset that may not be immediately obvious.
# How it works:

# Feature Extraction: Convert documents into a numerical format (vectors) that can be processed by clustering algorithms. This is typically done using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
# Clustering Algorithm: Apply a clustering algorithm to the document vectors to group similar documents together. Common algorithms include K-Means, Hierarchical Clustering, and DBSCAN.
# Steps to Implement Document Clustering
# Convert Documents to Vectors:
# Use TF-IDF to convert the text documents into numerical vectors.

# Apply Clustering Algorithm:
# Use a clustering algorithm like K-Means to group the document vectors into clusters.

# Analyze and Use Clusters:
# The resulting clusters can be analyzed to understand the topics they represent or used to improve search and retrieval.


# Step 2: Cluster document vectors
