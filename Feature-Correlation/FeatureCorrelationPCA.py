from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
import numpy as np
import random
import matplotlib.pyplot as plt


def princomp(A):

 M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = linalg.eig(cov(M)) # attention:not always sorted
 score = dot(coeff.T,M) # projection of the data in the new space
 return coeff,score,latent


def sample(DataBase,Size):
    Sample=[]
    #Need this to avoid repetition without removing items from the DataBase.
    CopiedDataBase=[]
    CopiedDataBase.extend(DataBase)
    #Create a list of picked images to prevent repetition
    #first lets have a list of all the features of all picked images


    for i in range(0,Size):
        image=random.randint(0,len(CopiedDataBase)-1)

        Sample.append(CopiedDataBase[image])
        #avoid repetition
        del CopiedDataBase[image]
    return np.asarray(Sample)






Means =[]
Errors=[]
SampleSizes=[]

populationPC=np.load('/data/placesFeatures/PCA/coefficientsFC7.npy')
population=np.load('/data/placesFeatures/batches/placesFC7.npy')


#To compute correlation I need the population principal components as a 1D array
populationPC1D=[]
for i in populationPC:
    populationPC1D.extend(i)

for SampleSize in range(10,len(population),10):
    x=[]
    SampleSizes.append(SampleSize)
    sample=sample(population,SampleSize)

    samplePC=princomp(sample)[0]
    #I need the Principal components in a 1D array to compute correlation
    samplePC1D=[]
    for i in samplePC:
         samplePC1D.extend(i)

    populationPC1D=np.asarray(populationPC1D)
    samplePC1D =np.asarray(samplePC1D)

    x.extend(np.pearsonr(populationPC1D,samplePC1D))
    Means.append(x[0])
    Errors.append(x[1])
plt.errorbar(SampleSizes,Means,Errors,linestyle='None', marker='^')
plt.show()
