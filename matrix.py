import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import codecs
import operator
from functools import reduce


def importJson():
    with codecs.open('./data/final.json','r','utf-8-sig') as json_file:
        return json.load(json_file)


if __name__ == '__main__':
    data           = importJson()
    producers      = list(map(lambda x:x['producer'],data)) 
    files          = list(map(lambda x:x['files'],data))
    topics         = list(map(lambda x:x['topic'],data))
    getNodeId      = lambda x:x['node_id']
    maxTopicId     = max(reduce(operator.concat,list(map(lambda x:  list(map(getNodeId,x)) ,topics))))
    maxProducerId  = max(list(map(getNodeId,producers)))
    maxFileId      = max(set(reduce(lambda x,y:x+y,files)))
    maxId          = max([maxProducerId,maxFileId,maxTopicId])
    A              = np.zeros((maxId+1,maxId+1))
    for d in data:
        producer = d['producer']
        files    = d['files']
        topics   = d['topic']
        zipped   = zip(files,topics)
        x        = producer['node_id']
        for (fnid,topic) in zipped:
            tnid = topic['node_id']
            #if(producer['role']=='Alpha'):
            #    A[x][fnid] = 5
             #   A[fnid][x] = 5
            #elif(producer['role']=='Beta'):
            #    A[x][fnid] = 3
            #    A[fnid][x] = 3 
            #else:
            #    A[x][fnid] = 1
            #    A[fnid][x] = 1 
            A[x][fnid]    = 1
            A[fnid][x]    = 1
            A[fnid][tnid] = 1
            A[tnid][fnid] = 1
    #plt.imshow(np.random.random((10,10)))
    #plt.imshow(A)
    #plt.colorbar()
    #plt.show()
    #plt.show()

    np.save('./data/am.npy',A)
