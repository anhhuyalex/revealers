#This code is meant to be used with actual data (not just random features)
import random
from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

def sample2(DataBase,Size):
    Sample=[]
    Vector=[]
    #Need this to avoid repetition without removing items from the DataBase.
    CopiedDataBase=[]
    CopiedDataBase.extend(DataBase)
    #Create a list of picked images to prevent repetition
    #I will need to know how many features I have per image
    Feats= len(DataBase[0])
    #first lets have a list of all the features of all picked images
    for i in range(0,Size):
        image=random.randint(0,len(CopiedDataBase)-1)

        Sample.extend(CopiedDataBase[image])
        #avoid repetition
        del CopiedDataBase[image]
    #Now I have all features of all images, but they are not in the right order (feature 1 of image 2 is in position 11 for 10 features)
    #Ultimately I want an array (Vector), where the first number is the average of the first feature for all sampled images.
    for f in range(0,Feats):
        #I want to get the average of each feature across sampled images
        average=0.0
        for j in range(0,Size):
            #In the matrix, feature 1 appears on positions Feats(j)+1 (first number of each image)
            average=average+Sample[Feats*j+f]
        average=average/Size
        Vector.append(average)
    return Vector



#This function will receive the DataBase and a sample size.
#It returns the average correlation and error, doing 10 repetitions of the function sample.
def CorrelationForSamplesize(DataBase,Size,):
    Result=[]

    #Repeat experiment 10 times
    for i in range(0,10):
        Result.append(pearsonr(sample2(DataBase,Size),sample2(DataBase,Size))[0])
    Average = sum(Result)/float(len(Result))
    Error = np.std(Result)
    Result=[]

     #At the end we return the data in a list
    Result.append(Average)
    Result.append(Error)
    return Result

#this has 50 batches of 10 images
Data=np.load('/home/ray/caffe/_temp/batches/batch39/places_200000_fc7.npy')

#This will have 500 images
Data.flatten()



Means =[]
Errors=[]
SampleSizes=[]

for i in range(1,len(Data),10):
    x=[]
    SampleSizes.append(i)
    x.extend(CorrelationForSamplesize(Data,i))
    Means.append(x[0])
    Errors.append(x[1])
plt.errorbar(SampleSizes,Means,Errors,linestyle='None', marker='^')
plt.show()
