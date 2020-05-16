import re
import numpy as np
import pandas as pd
from pprint import pprint

from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import csv

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#Import Data
df = pd.read_csv("mashup.lda.csv")
df2 = pd.read_csv("../apis.lda.csv")
#print(df.name.unique())

#Convert to List
data = df.desc.values.tolist()
data2 = df2.desc.values.tolist()
apis = df.apis.values.tolist()
apis_name_ = df2.name.values.tolist()
csvwriter_apis = csv.writer(open("apis.csv","w"))
csvwriter_apis_names = csv.writer(open("../apis.names.csv","w"))
for i in apis:
    csvwriter_apis.writerow([i])
for i in apis_name_:
    csvwriter_apis_names.writerow([i])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

for item in data2:
    data.append(item)

data_words = list(sent_to_words(data))

#print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=20) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=20)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(len(data_lemmatized))
print(data_lemmatized[:1])

corpus = []
for word_lst in data_lemmatized:
    t = ""
    for w in word_lst:
        t+=w
        t+=" "
    corpus.append(t)

vectorizer = CountVectorizer() 

X = vectorizer.fit_transform(corpus)

# 获取词袋模型中的所有词语   
word = vectorizer.get_feature_names()  
print(len(word)) 

# 获取每个词在该行（文档）中出现的次数
counts =  X.toarray()
#print (counts[0])

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
#tfidf = transformer.fit_transform(counts) #与上一行的效果完全一样
#print(tfidf)
print(tfidf.toarray())
tfidf_array = tfidf.toarray()
print(len(tfidf_array))


