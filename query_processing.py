from preprocessing import preprocess

def process_query(query, vectorizer):
    processed_query = preprocess(query)
    query_vec = vectorizer.transform([processed_query])
    return query_vec
