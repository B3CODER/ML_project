import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

df =pd.read_csv('papers.csv')

stop_words =set(stopwords.words('english'))

new_words =['fig', 'figure', 'image', 'sample', 'using',
            'show', 'result', 'large', 'also', 'one', 'two',
            'three', 'four', 'five', 'six' , 'seven' , 'eight',
            'nine']

stop_words = list(stop_words.union(new_words))

def pre_process(text):
    
    text = text.lower()
    text =re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    text =re.sub("(\\d|\\W)+"," ",text)
    text =text.split()
    
    text =[word for word in text if word not in stop_words]
    text =[word for word in text if len(word)>=3]
    
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]
    return ' '.join(text)

docs =df['paper_text'].apply(lambda x:pre_process)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df = 0.95,
                     max_features = 10000,
                     ngram_range=(1,3))

word_countvector = cv.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer  = TfidfTransformer(smooth_idf=True , use_idf=True)
tfidf_transformer.fit(word_countvector)


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col , coo_matrix.data)
    return sorted(tuples ,key =lambda x: (x[1] ,x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

# get feature names
feature_names=cv.get_feature_names()

def get_keywords(idx, docs):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return keywords

def print_results(idx,keywords, df):
    # now print the results
    print("\n=====Title=====")
    print(df['title'][idx])
    print("\n=====Abstract=====")
    print(df['abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])
idx=941
keywords=get_keywords(idx, docs)
print_results(idx,keywords, df)

    