import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import os
import string
import pickle
import re
import xlrd

from gensim import corpora, models, similarities
import logging

f=open('dictionary.dic','r')  
dictionary=pickle.load(f)  
f.close()  
f=open('lsi.lsi','r')  
lsi=pickle.load(f)  
f.close() 

a = open('out.txt','r') .read()
apiNum = [[i for i in api.split(' ') if  not(i == '')] for api in a.split('\n')]

apidir = '/home/pengqianyang/nlp/nlp/api.xlsx'
apidata = xlrd.open_workbook(apidir)
apitable = apidata.sheet_by_index(0)

output1 =open('output1.txt','w')  
output2 =open('output2.txt','w')  
output3 =open('output3.txt','w')  

for i in range(0,200):
    dir = "/home/pengqianyang/nlp/nlp/result/"
    file = str(i)+".txt"
    if os.path.exists(dir+file):
        raw_text = open(dir+file,'r').read()
        print>> output1,file+':'
        print>> output2,file+':'
        print>> output3,file+':'
        pattern = re.compile(r'text _!(.*?)_!',re.M)
        match = pattern.findall(raw_text)
        raw_material = [line.strip() for line in match]
        texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in raw_material]

        english_stopwords = stopwords.words('english')
        texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
        texts_filtered = [[word for word in document if not word in string.punctuation] for document in texts_filtered_stopwords]

        st = LancasterStemmer()
        texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]
        print texts_stemmed
        #for i in range(len(texts_filtered)):
        #    if texts_filtered[i] != texts_stemmed[i]:
        #        print texts_filtered[i] + ' --- ' + texts_stemmed[i]
        texts = texts_stemmed
        corpus = [dictionary.doc2bow(text) for text in texts]
        index = similarities.MatrixSimilarity(lsi[corpus])
        #------------------------------------------------------------------------------------------------------------------------------------------
        for kapi in apiNum[i]:
            print>> output1,apitable.row_values(int(kapi))[2]
            target =  unicode(apitable.row_values(int(kapi))[2]).strip()
            print>> output1,target
            raw_material = target.strip()
            texts_tokenized = [word.lower() for word in word_tokenize(raw_material)]

            english_stopwords = stopwords.words('english')
            texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]

            texts_filtered = [word for word in texts_filtered_stopwords if not word in string.punctuation] 

            st = LancasterStemmer()
            texts_stemmed = [st.stem(word) for word in texts_filtered]

            #for i in range(len(texts_filtered)):
            #    if texts_filtered[i] != texts_stemmed[i]:
            #        print texts_filtered[i] + ' --- ' + texts_stemmed[i]

            text = texts_stemmed

            ml_bow = dictionary.doc2bow(text)
            ml_lsi = lsi[ml_bow]
            sims = index[ml_lsi]
            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

            print>> output1,sort_sims
            print >> output1,'\n'
        for kapi in apiNum[i]:
            print >> output2,apitable.row_values(int(kapi))[2]
            target =  unicode(apitable.row_values(int(kapi))[3]).strip()
            print >> output2,target
            raw_material = target.strip()
            texts_tokenized = [word.lower() for word in word_tokenize(raw_material)]

            english_stopwords = stopwords.words('english')
            texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]

            texts_filtered = [word for word in texts_filtered_stopwords if not word in string.punctuation] 

            st = LancasterStemmer()
            texts_stemmed = [st.stem(word) for word in texts_filtered]
            print >> output2,texts_stemmed

            #for i in range(len(texts_filtered)):
            #    if texts_filtered[i] != texts_stemmed[i]:
            #        print texts_filtered[i] + ' --- ' + texts_stemmed[i]

            text = texts_stemmed

            ml_bow = dictionary.doc2bow(text)
            ml_lsi = lsi[ml_bow]
            sims = index[ml_lsi]
            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

            print >> output2,sort_sims
            print >> output2,'\n'
        for kapi in apiNum[i]:
            print>> output3, apitable.row_values(int(kapi))[2]
            target =  unicode(apitable.row_values(int(kapi))[8]).strip()
            raw_material = target.strip()
            texts_tokenized = [word.lower() for word in word_tokenize(raw_material)]

            english_stopwords = stopwords.words('english')
            texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]

            texts_filtered = [word for word in texts_filtered_stopwords if not word in string.punctuation] 

            st = LancasterStemmer()
            texts_stemmed = [st.stem(word) for word in texts_filtered]
            print>> output3, texts_stemmed

            #for i in range(len(texts_filtered)):
            #    if texts_filtered[i] != texts_stemmed[i]:
            #        print texts_filtered[i] + ' --- ' + texts_stemmed[i]

            text = texts_stemmed

            ml_bow = dictionary.doc2bow(text)
            ml_lsi = lsi[ml_bow]
            sims = index[ml_lsi]
            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

            print >> output3,sort_sims
            print >> output3,'\n'