import random
from scipy.stats.stats import pearsonr
import numpy
import matplotlib.pyplot as plt

def CreateData() :
    #Creates a matrix of Image number and features.
    #Images = int(raw_input("Enter the number of images:"))
    #Features = int(raw_input("Enter the number of features:"))
    Images = 20000
    Features = 10
    Matrix = [[random.randint(0, 10) for i in xrange(Features)] for i in xrange(Images)]
    return Matrix


#Takes samples of a given size out of a matrix and gives you the vector with averaged features.

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
    Error = numpy.std(Result)
    Result=[]

     #At the end we return the data in a list
    Result.append(Average)
    Result.append(Error)
    return Result




DataBase=CreateData()

Means=[]
Errors=[]
SampleSizes=[]

for i in range(1,len(DataBase),1000):
    x=[]
    SampleSizes.append(i)
    x.extend(CorrelationForSamplesize(DataBase,i))
    Means.append(x[0])
    Errors.append(x[1])
plt.errorbar(SampleSizes,Means,Errors,linestyle='None', marker='^')
plt.show()
