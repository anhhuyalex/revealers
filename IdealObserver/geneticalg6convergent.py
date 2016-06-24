
# coding: utf-8

# In[1]:

import random
from scipy.stats import pearsonr
import scipy.io as sio
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
import time


# In[2]:

mat_contents = sio.loadmat('verySmallWavelets2.mat')


# In[3]:

success=False
popsize = 100
corGoal =.99
pCombine= .4
pMutate= 0.5
pMigrate= 0.6
rejectcor = 0.5


# In[4]:

waveletFeatures = mat_contents['waveletFeatures']
scores = mat_contents['scores']
wavecoef = mat_contents['wavecoef']
filters = mat_contents['filters']
meanwavefeatures = np.squeeze(mat_contents['meanwavefeatures'])

walk_stdev = np.std(scores,axis=0)*.05
Matrix=[]

scoresstd = np.std(scores,axis=0)
scoresmean = np.mean(scores,axis=0)
#scoresstd contains the means of the scores
#scoresmean contains the stds of the scores


# In[5]:

"""
print wavecoef.shape, 'wavecoef'
print meanwavefeatures.shape, 'meanwavefeatures'
print filters.shape, 'filters'
print scores.shape, 'scores'
print scoresmean.shape, 'scoresmean'
print waveletFeatures.shape, 'waveletFeatures'
"""


# In[6]:

def CreateImage(scoresmean,scoresstd,wavecoef,meanwavefeatures,thisPCA=None):
    if thisPCA == None:
        thisPCA = []
        for i in range(900):
            thisPCA.append(np.random.normal(scoresmean[i],scoresstd[i]))
    features = wavecoef[:,0:900].dot(thisPCA)+meanwavefeatures
    thisImage = features.dot(filters.T)
    return thisImage


# In[7]:

def FirstMatrix(popsize,Target,rejectcor):
    Matrix = []
    pca_list = []
    while len(Matrix) < popsize:
        thisPCA = []
        for i in range(900):
            thisPCA.append(np.random.normal(scoresmean[i],scoresstd[i]))
        image = CreateImage(scoresmean,scoresstd,wavecoef,meanwavefeatures,thisPCA)
        r = pearsonr(image,Target)[0]
        if r < rejectcor:
            Matrix.append(image)
            pca_list.append(thisPCA)
    return np.array(Matrix),np.array(pca_list)


# In[25]:

Target = waveletFeatures[:,3812].dot(filters.T)
#Target = CreateImage(scoresmean,scoresstd,wavecoef,meanwavefeatures)
Best = CreateImage(scoresmean,scoresstd,wavecoef,meanwavefeatures)
thismatrix,pca_list = FirstMatrix(popsize,Target,rejectcor)


# In[11]:

#thismatrix.shape


# In[13]:

def pair_indices(popsize):
    indexmatrix = range(popsize)
    random.shuffle(indexmatrix)
    ListOfPairs=[]
    for a in range(0,50):
        even = 2*a
        for k in range(0,5):
            odd=even+k*20+1
            if odd>99:
                odd=odd-100
            Pair=[indexmatrix[even],indexmatrix[odd]]
            ListOfPairs.append(Pair[:])

    return np.array(ListOfPairs)


# In[14]:

def compare_images(indexpairs,imagematrix,pca_list,Target,Best):
    imagebag = []
    corrarray = []
    #calculate correlations
    for image in imagematrix:
        corr = pearsonr(image,Target)[0]
        corrarray.append(corr)
    corrarray = np.array(corrarray)
    #calculate best correlation in this generation
    maxcorr = np.amax(corrarray)
    #calculate average correlation in this generation
    avgcorr = np.mean(corrarray)
    #check if best correlation this gen is better than overall best
    if maxcorr > pearsonr(Best,Target)[0]:
        maxindex = np.argmax(corrarray)
        Best = np.array(imagematrix[maxindex])
    #Let the Hunger Games begin
    for i in indexpairs:
        corr1 = corrarray[i[0]]
        corr2 = corrarray[i[1]]
        if corr1 > corr2:
            imagebag.append(i[0])
        else:
            imagebag.append(i[1])
    #survivors
    newindices = np.array(random.sample(imagebag,100))
    pca_list_nextgen = pca_list[newindices]
    return pca_list_nextgen,maxcorr,avgcorr,Best


# In[57]:

def Evolve(pca_list,popsize,pCombine,pMutate,pMigrate):
    #Combinations
    for i in range(popsize):
        if random.random()<pCombine:
            parent1pcs = pca_list[i]
            parent2pcs = pca_list[random.randint(0,popsize-1)]
            parent1pcs[np.arange(1,900,2)]=parent2pcs[np.arange(1,900,2)]
            pca_list[i] = parent1pcs

    #Mutate
    for i in range(popsize):
        if random.random() < pMutate:
            thisPCA = pca_list[i]
            for i,item in enumerate(thisPCA):
                thisPCA[i] = np.random.normal(item,walk_stdev[i])

    #Migration
    for i in range(popsize):
        if random.random()<pMigrate:
            thisPCA = []
            for j in range(900):
                thisPCA.append(np.random.normal(scoresmean[j],scoresstd[j]))
            pca_list[i] = np.array(thisPCA)

    return pca_list










# In[60]:

newmatrix = np.array(thismatrix)
newpca_list = np.array(pca_list)
Best = CreateImage(scoresmean,scoresstd,wavecoef,meanwavefeatures)

while not success:
    print pearsonr(Best,Target)
    if pearsonr(Best,Target)[0]>corGoal:
        success = True
    tic = time.time()
    indexpairs = pair_indices(popsize)
    newpca_list,maxcorr,avgcorr,Best = compare_images(indexpairs,newmatrix,newpca_list,Target,Best)
    print "Best correlation of this generation is ", maxcorr
    print "Average correlation of this generation is ", avgcorr
    newpca_list = Evolve(newpca_list,popsize,pCombine,pMutate,pMigrate)
    for i in range(len(newmatrix)):
        newmatrix[i] = CreateImage(scoresmean,scoresstd,wavecoef,meanwavefeatures,newpca_list[i])

    toc = time.time()
    pMigrate *= 0.5
    print "This loop took %f s" %(toc-tic)
