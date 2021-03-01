import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import codecs
import operator
from functools import reduce

scores = {
        'AlphaAlpha':9,
        'AlphaBeta':8,
        'AlphaGamma':7,
        'BetaAlpha': 6,
        'BetaBeta': 5,
        'BetaGamma': 4,
        'GammaAlpha': 3,
        'GammaBeta': 2,
        'GammaGamma': 1,
}

def importJson():
    with codecs.open('./data/r2.json','r','utf-8-sig') as json_file:
        return json.load(json_file)


if __name__ == '__main__':
    data           = importJson()
    producers      = list(map(lambda x:x['producer'],data)) 
    #files          = list(map(lambda x:x['files'],data))
    topics         = list(map(lambda x:x['topics'],data))
    def getNodeId(x):
        print(x)
        return x['node_id']
    producersIds   = set(map(getNodeId,producers))
    topicsIds      = list(map(lambda x: list(map(getNodeId,x)),topics))
    topicsIds      = set(reduce(operator.concat,topicsIds))
    print(topicsIds)
    #maxTopicId     = max(reduce(operator.concat,list(map(lambda x:  list(map(getNodeId,x)) ,topics))))
    #maxProducerId  = max(list(map(getNodeId,producers)))
    #maxFileId      = max(set(reduce(lambda x,y:x+y,files)))
   # maxId          = max([
    #    maxProducerId,
        #maxFileId,
     #   maxTopicId])
    maxNodes       = len(producersIds|topicsIds) 
    #A              = np.zeros((maxId+1,maxId+1))
    A = np.zeros((maxNodes+1,maxNodes+1))
    for d in data:
        producer = d['producer']
        #files    = d['files']
        topics   = d['topics']
        #zipped   = zip(files,topics)
        zipped   = zip(range(len(topics)),topics)
        pnid     = producer['node_id'] % maxNodes
        prole    = producer['role']
        
        for (_,topic) in zipped:
            tnid  = topic['node_id'] % maxNodes
            trole = topic['role'] 
            key = '{}{}'.format(prole,trole)
            A[pnid][tnid] = scores[key]
            A[tnid][pnid] = scores[key]
    #plt.imshow(np.random.random((10,10)))
    #plt.imshow(A)
    #plt.colorbar()
    #plt.show()
    #plt.show()

    np.save('./data/am1.npy',A)
