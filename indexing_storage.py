import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets
from preprocessing import preprocess

vectorizer_path = './wikir_100'
document_vectors_path = 'wikir_100.joblib'
documents_path = 'docs.jolib_100'

def build_or_load_index(dataset_name='cranfield', num_docs=None):
    os.environ['IR_DATASETS_HOME'] = '/dataset'
    dataset = ir_datasets.load(dataset_name)
    documents = []

    if os.path.exists(vectorizer_path) and os.path.exists(document_vectors_path) and os.path.exists(documents_path):
        vectorizer = joblib.load(vectorizer_path)
        document_vectors = joblib.load(document_vectors_path)
        documents = joblib.load(documents_path)
    else:
        # max_df parameter is used to exclude terms that appear too frequently in the corpus, which are likely to be non-informative (e.g., common words across all documents).
        vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english')
        processed_texts = []
        
        for i, doc in enumerate(dataset.docs_iter()):
            if num_docs is not None and i >= num_docs:
                break
            processed_text = preprocess(doc.text) 
            processed_texts.append(processed_text)

        document_vectors = vectorizer.fit_transform(processed_texts)
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(processed_texts, documents_path)
        joblib.dump(document_vectors, document_vectors_path)
        documents = processed_texts
        
        
    return vectorizer, document_vectors, documents

def load_data(dataset_name='cranfield'):
    dataset = ir_datasets.load(dataset_name)
    documents = []
    for i, doc in enumerate(dataset.docs_iter()):
        documents.append(doc.text) 
    
    return documents
