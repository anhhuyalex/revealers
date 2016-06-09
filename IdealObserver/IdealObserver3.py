import random
from scipy.stats.stats import pearsonr
import scipy.io as sio
import numpy as np

global success
global ImageSize
global corGoal
global pCombine
global pMigrate
global pMutate
success=False
corGoal =.9
pCombine= .4
pMutate= 0.5
pMigrate= 0.6

mat_contents = sio.loadmat('verySmallWavelets2.mat')
waveletFeatures = mat_contents['waveletFeatures']
scores = mat_contents['scores']
np.asarray(scores)
wavecoef=mat_contents['wavecoef']
np.asarray(wavecoef)
filters=mat_contents['filters']
meanwavefeatures=mat_contents['meanwavefeatures']
np.asarray(meanwavefeatures)
meanwavefeatures = np.squeeze(np.asarray(meanwavefeatures))

walk_stdev = np.std(scores)*.05
Matrix=[]



scores2=[[],[]]
for i in xrange(900):
    scores2[0].append(np.mean(scores.conj().T[i][:]))
    scores2[1].append(np.std(scores.conj().T[i][:]))

#scores2[0] contains the means of the scores
#scores2[1] contains the stds of the scores

def CreateImage():
    thisPCA=[]
    thisIm=[]
    for l in xrange(900):
            thisPCA.append(np.random.normal(scores2[0][l],scores2[1][l]))
    thisPCA = np.asarray(thisPCA)
    features = (wavecoef[:,0:900]).dot(thisPCA.conj().T)[:]+meanwavefeatures
    thisPixel=features.conj().T.dot(filters.conj().T)[:]
    thisIm.append(thisPixel[:])
    thisIm.append(thisPCA[:])
    return thisIm



def FirstMatrix():
    #global MatrixPixels

    for x in xrange(100):
        thisIm=CreateImage()
        Matrix.append(thisIm[:])
    #MatrixPixels=np.asarray(MatrixPixels)


def Pair(Matrix):
    random.shuffle(Matrix)
    ListOfPairs=[]
    for a in range(0,49):
        even = 2*a
        for k in range(0,5):
            odd=even+k*20+1
            if odd>99:
                odd=odd-100
            Pair=[Matrix[even][:],Matrix[odd][:]]
            ListOfPairs.append(Pair[:])

    return ListOfPairs


def Compare(ListOfPairs,Target,Matrix):
    global success
    global Best
    ImageBag=[]

    for i in (ListOfPairs):

        if pearsonr(i[0][0],Target[0])[0]>pearsonr(i[1][0],Target[0])[0]:
            ImageBag.append(i[0][:])

            if pearsonr(i[0][0],Target[0])[0]>pearsonr(Best[0],Target[0])[0]: Best=i[0][:]
        else:
            ImageBag.append(i[1][:])

            if pearsonr(i[1][0],Target[0])[0]>pearsonr(Best[0],Target[0])[0]: Best=i[1][:]
    return random.sample(ImageBag,100)


def Evolve():

    for i in xrange(len(Matrix)):
        #We do combinations on PCA
        if random.random()<pCombine:
            parent2=[]
            parent2.extend(Matrix[random.randint(0,len(Matrix)-1)][1][:])
            #combine PCA with parent2


            for p in range(1,len(Matrix[i][1]),2):
                Matrix[i][1][p]=parent2[p]
            #Now we need to update the pixels for the new PCA
            thisPCA=Matrix[i][1][:]
            features = (wavecoef[:,0:900]).dot(thisPCA.conj().T)[:]+meanwavefeatures
            thisPixel=features.conj().T.dot(filters.conj().T)[:]
            Matrix[i][0] = thisPixel[:]
    for i in xrange(len(Matrix)):
        #We do mutations on PCA
        if random.random()<pMutate:
            for x in xrange(len(Matrix[i][1])):
                #Matrix[i][1][x]=Matrix[i][1][x]+random.uniform(-1,1)
                Matrix[i][1][x]=np.random.normal(Matrix[i][1][x],walk_stdev)

            #Now we update pixels:
            thisPCA=Matrix[i][1][:]
            features = (wavecoef[:,0:900]).dot(thisPCA.conj().T)+meanwavefeatures
            thisPixels=features.conj().T.dot(filters.conj().T)[:]
            Matrix[i][0] = thisPixels[:]
    for i in xrange(len(Matrix)):
        #We do migration
        if random.random()<pMigrate:
            Matrix[i]=CreateImage()




FirstMatrix()
print 'matrix created'
Target = CreateImage()
Best = CreateImage()



while success==False:
    if pearsonr(Best[0],Target[0])[0]>=corGoal:
        success=True
        break

    ListOfPairs=Pair(Matrix)
    Matrix=Compare(ListOfPairs,Target,Matrix)[:]
    Evolve()
    pMigrate=pMigrate/2
    print 'best correlation ',pearsonr(Best[0],Target[0])[0]

    #print Best
#print Best
#print Target
print 'best correlation', pearsonr(Best[0],Target[0])
print success

