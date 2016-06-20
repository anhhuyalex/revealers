import caffe
import scipy
import os
import numpy as np

FinalArray = []

dir = '/data/placesFeatures/batches'
for root, dirs, filenames in os.walk(dir):
    for f in filenames:
        path = dir + f +'places_200000_fc7.npy'
        Data=[]
        Data=np.load(path)
        #We make a conversion so that we have a list of all images. For fc7, we originally have a list of batches of 10

        Data2=[]
        for i in Data:
            Data2.extend(i)
        FinalArray.extend(Data2)

np.save('/data/placesFeatures/batches/FC7.npy',FinalArray)
