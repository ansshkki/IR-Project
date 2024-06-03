import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
# Query Refinement (Query Formulation Assistance, Query Suggestions)
# Purpose: Enhancing the user's query by expanding it with synonyms to improve retrieval performance.
def expand_query(query):
    tokens = word_tokenize(query)
    expanded_tokens = []
    for token in tokens:
        synonyms = wordnet.synsets(token)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_tokens.append(lemma.name())
    return ' '.join(set(expanded_tokens))


# Example
# expand_query("good")
# output unspoilt ripe adept safe commodity trade_good near just respectable skilful dependable upright undecomposed effective expert unspoiled full right in_effect thoroughly soundly serious secure sound well salutary in_force good beneficial proficient goodness estimable dear practiced honorable honest skillful




# Query Refinement (Query Formulation Assistance, Query Suggestions)
# Purpose:
# Query refinement involves improving or expanding the user's search query to retrieve more relevant documents. The goal is to help the user formulate a better query, which can result in more accurate search results. This can include suggesting alternative queries or expanding the original query with synonyms.
# What it does:
# Synonym Expansion: When a user submits a query, it might be too specific or might not cover all relevant synonyms. By expanding the query with synonyms, you increase the chances of retrieving documents that are relevant but use different terminology.

# Query Suggestions: Based on the initial query, the system can suggest alternative queries that might yield better results. This can be based on popular queries or common modifications.

# Example:

# User's original query: "during his time in t he held the posts of cabinet secretary chairman of"
# Expanded query: "during his time in t he held the posts of cabinet secretary chairman of time period serve office role chief head"
# The expanded query now includes synonyms for "time" (e.g., "period"), "held" (e.g., "serve"), "posts" (e.g., "office"), etc. This helps retrieve documents that might use different words but have the same meaning.