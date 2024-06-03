import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    # Stemming reduces words to their root or base form, which helps in reducing the vocabulary size and improving model generalization by treating different forms of the same word as identical.

    # word.isalnum(): Checks if the token contains only alphanumeric characters (letters or digits). This condition filters out punctuation and other non-alphanumeric characters, keeping only actual words.

    # word not in stop_words: Checks if the token is not in a list of stop words. Stop words are common words that are often removed from text because they typically do not carry much meaning and can be ignored without loss of information. This condition removes stop words from the list of tokens.
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)
