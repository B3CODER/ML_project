from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (words) and document names
feature_names = vectorizer.get_feature_names_out()
document_names = [f"Document {i+1}" for i in range(len(documents))]

# Create a DataFrame for better visualization
import pandas as pd
df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), index=document_names, columns=feature_names)

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(df_tfidf)
