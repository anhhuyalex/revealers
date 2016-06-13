import caffe
import scipy
import os
import numpy as np

dir = '/home/ray/caffe/data/smallPlaces/smallPlaces'

for root, dirs, filenames in os.walk(dir):
    for f in filenames:
        path = dir + f
        Image = caffe.io.load_image(path)
        resized = caffe.io.resize_image(Image,(227,227))
        newpath = '/home/ray/caffe/data/smallPlaces/resizedSmallPlaces'+f
        np.save(newpath,resized)

