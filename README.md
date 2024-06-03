# Information Retrieval Project with Document Clustering and Topic Detection and Expand query


## Overview

This project implements a sophisticated search engine with features like document clustering and topic detection to improve search accuracy and relevance. The core components include preprocessing, query processing, indexing, document retrieval, clustering, and topic detection.


## Project Structure

- main.py
This is the entry point of the application.
SearchEngine: Initializes the search engine.
search endpoint: Accepts search queries and parameters, processes them through the search engine, and returns the results.

- search_engine.py
The main class that integrates all components to perform search operations.
SearchEngine: Main class orchestrating the search process.
google_search_engine: Method to perform the search, including query expansion, document ranking, optional clustering, and topic detection.

- query_processing.py
Handles preprocessing and vectorization of search queries.
process_query: Preprocesses the query and converts it into a vector using a provided vectorizer.

- retrieval.py
Responsible for building or loading document indices and vectors.
build_or_load_index: Creates or loads the document vectorizer and document vectors.
load_data: Loads raw document data.

- indexing_storage.py
Contains utility functions for query expansion and storage.
expand_query: Expands the user query by adding synonyms to improve retrieval.

- query_refinement.py
Refines the search query and ranks the documents based on similarity.
rank_documents: Ranks documents by computing the cosine similarity between the query vector and document vectors.

- document_clustering.py
Performs document clustering to group similar documents.
find_optimal_num_clusters: Uses the Elbow Method to determine the optimal number of clusters.
cluster_documents: Clusters the document vectors.
find_relevant_doc_vector_by_clusters: Identifies relevant document vectors by comparing query vectors to cluster centers.

## How the Project Works

1. Preprocess and Vectorize Documents:

- Load and preprocess the documents.
- Convert documents to vectors using TF-IDF.

2. Document Clustering:
- Cluster the document vectors using KMeans.
- Determine the optimal number of clusters using the Elbow Method.

3. Query Processing:
- Expand and preprocess the user query.
- Convert the query to a vector using the same vectorizer used for documents.

4. Document Ranking:
- Rank the documents based on cosine similarity - between the query vector and document vectors.
Optionally, refine the search by only considering documents in relevant clusters.

5. Topic Detection:
- Apply LDA to the top documents to detect underlying topics.
- Print and return the top words for each detected topic.


## Dependencies
- nltk: For natural language processing tasks.
- scikit-learn: For vectorization, clustering, - and similarity computations.
- ir_datasets: For loading datasets.
- joblib: For saving and loading models.
- uvicorn: For running the FastAPI application.
- FastAPI: For creating the API endpoints.
- gensim: For LDA topic modeling (if needed for - more advanced topic detection).