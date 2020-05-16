import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist
import numpy as np
from scipy import stats

csvreader1 = csv.reader(open("apis.csv","r"))
csvreader2 = csv.reader(open("topic.vector.csv","r"))
csvreader3 = csv.reader(open("../apis.names.csv","r"))

apis = []
apis_set = {}
cnt_single = 0
for line in csvreader1:
    l = line[0].split(", ")
    if(len(l)==1):
        cnt_single+=1
    tmp = {}
    for i in l:
        tmp[i] = 1
        if i not in apis_set:
            apis_set[i] = 0
        apis_set[i]+=1    
    apis.append(tmp)
print("apis total: ",len(apis_set))
print("mashup total: ",len(apis))
print("cnt_single: ",cnt_single)

ind = 0
stat = 0
filter_lst = {}
for lst in apis:
    flag = False
    for i in lst:
        if apis_set[i]==1:
            flag = True
            break
    if flag:
        stat += 1
        filter_lst[ind] = 1
    ind+=1    
print(stat)

    

lda_vectors = []
apis_lda_vectors = []
for line in csvreader2:
    tmp = []
    for i in range(1,len(line)):
        tmp.append(float(line[i]))
    if len(lda_vectors) < len(apis):
        lda_vectors.append(tmp)
    else:
        apis_lda_vectors.append(tmp)

apis_name = []
for line in csvreader3:
    apis_name.append(line[0])    
#print(apis_name)
print(len(apis_name),len(apis_lda_vectors))

#print(cosine(lda_vectors[0], lda_vectors[1]))

def apis_sim(apis1,apis2):
    hit = 0
    for k in apis1.keys():
        if k in apis2:
            hit+=1
    merge = len(apis1)+len(apis2)-hit
    hit = float(hit)
    sim1 = hit/min(len(apis1),len(apis2))
    sim2 = hit/max(len(apis1),len(apis2))
    sim3 = hit/merge
    sim4 = hit/len(apis1)
    sim5 = hit/len(apis2)
    return [sim1,sim2,sim3,sim4,sim5]

def distance(vector1,vector2):
    sim1 = cosine(vector1, vector2)
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    distance2=np.sqrt(np.sum(np.square(vector1-vector2)))
    return [sim1,distance2]


#print(apis_sim(apis[0],apis[1]))
#csvwriter = csv.writer(open("sim.distance.1000.csv","w"))
#csvwriter = csv.writer(open("../sim.distance.apis.mashup.1000.csv","w"))
sims = [[],[],[],[],[]]
distances = [[],[]]
'''
for i in range(len(apis)):
    print(i)
    for j in range(i+1,len(apis)):
        tmp1 = apis_sim(apis[i],apis[j])
        for k in range(5):
            sims[k].append(tmp1[k])
        tmp2 = distance(lda_vectors[i],lda_vectors[j])
        for k in range(2):
            distances[k].append(tmp2[k])
        tmp1.append(tmp2[0])
        tmp1.append(tmp2[1])
        csvwriter.writerow(tmp1)

for sim in sims:
    for distance in distances:
        a = np.array(sim)
        b = np.array(distance)
        print(stats.pearsonr(a, b))

for i in range(len(apis)):
    print(i)
    distance_ = {}
    for j in range(len(apis_name)):
        name_ = apis_name[j]
        tmp = distance(lda_vectors[i],apis_lda_vectors[j])
        distance_[name_] = tmp[0]
    d = sorted(distance_.items(),key = lambda x:(x[1]))
    #print(d)
    lst = []
    for item in d:
        #print(item)
        lst.append(item[0])
    #print(lst)
    csvwriter.writerow(lst)
'''

csvreader = csv.reader(open("sim.distance.200.csv","r"))
for line in csvreader:
    for i in range(5):
        sims[i].append(float(line[i]))
    for i in range(2):
        distances[i].append(float(line[i+5]))

apis_recc = []
mashup_sims = []
distance_matrix = []
for i in range(len(apis)):
    apis_recc.append({})
    mashup_sims.append([])
    tmp = []
    for j in range(len(apis)):
        tmp.append(0)
    distance_matrix.append(tmp)


ind = 0
for i in range(len(apis)):
    for j in range(i+1,len(apis)): 
        mashup_sims[i].append((j,distances[0][ind]))
        mashup_sims[j].append((i,distances[0][ind]))
        distance_matrix[i][j] = distances[0][ind]
        distance_matrix[j][i] = distances[0][ind]
        ind+=1

hiit = 0
recc_amount = 500
recc_apis_amount = 200

accu = 0
recc_mashup = []
record = 0
amounts  = 0
csvwriter5 = csv.writer(open("recc_apis.200.csv","w")) 
for i in range(len(apis)):
    
    recc_mashup.append([])

    #if i in filter_lst:
    #    continue

    recc_apis = {}
    row = mashup_sims[i]
    row = sorted(row, key=lambda x: (x[1]))
    #if i==1:
        #print(row)
    ind = 0
    while(len(recc_mashup[i])<recc_amount and ind<len(row) and len(recc_apis)<recc_apis_amount):
        j = row[ind][0]
        s = row[ind][1]-0.3
        flag = True
        for k in recc_mashup[i]:
            if distance_matrix[k][j]<s:
                flag = False
                break
        if flag:
            recc_mashup[i].append(j)
            for k in apis[j]:
                if k not in recc_apis:
                    recc_apis[k] = 0
                recc_apis[k]+=1
                if(len(recc_apis)>=recc_apis_amount):
                    break    
        ind+=1
    if len(recc_mashup[i])<recc_amount:
        record+=1
    
    
   
    hit = 0
    for k in apis[i]:
        if k in recc_apis:
            hit+=1
    if hit==len(apis[i]):
        hiit+=1
    accu+=float(hit)/len(apis[i])
    amounts+=len(recc_apis)

    recc_lst = []
    recc_lst_sorted = sorted(recc_apis.items(),key = lambda item:item[1],reverse = True)
    for item in recc_lst_sorted:
        recc_lst.append(item[0])
    csvwriter5.writerow(recc_lst)
    
    
print(hiit)
print(accu/(len(apis)-len(filter_lst)))
print("mean_recc_amount", amounts/(len(apis)-len(filter_lst)))
print(record)
    


