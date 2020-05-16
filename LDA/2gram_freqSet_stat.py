import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist
import numpy as np
from scipy import stats
import json

csvreader1 = csv.reader(open("apis.csv","r"))
csvreader2 = csv.reader(open("topic.vector.csv","r"))

apis = []
apis_set = {}
for line in csvreader1:
    l = line[0].split(", ")
    tmp = {}
    for i in l:
        tmp[i] = 1
        apis_set[i] = {}
    apis.append(tmp)
print("apis total: ",len(apis_set))
print("mashup total: ",len(apis))

index = 0
for set_ in apis:
    print(index)
    index+=1
    l = list(set_.keys())
    sorted(l)
    for i in range(len(l)):
        key_i = l[i]
        for j in range(i+1,len(l)):
            key_j = l[j]
            if key_j not in apis_set[key_i]:
                apis_set[key_i][key_j] = 0
            apis_set[key_i][key_j]+=1

for key1,item in apis_set.items():
    for key2,val in item.items():
        if key1 not in apis_set[key2]:
            apis_set[key2][key1] = val

json.dump(apis_set,open("2gram.freqSet.stat.json","w"))

amount_stat = {}
for i in apis:
    l = len(i)
    if l not in amount_stat:
        amount_stat[l] = 0
    amount_stat[l]+=1
print(amount_stat)    
