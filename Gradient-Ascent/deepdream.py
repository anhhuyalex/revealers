from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe
from scipy.misc import imsave
#import googledream
import dream
img = np.float32(PIL.Image.open('../data/smallPlaces/resizedSmallPlaces/gsun_0a5c663d4ea492022362751c30b6476f.jpg'))

#print 'I am here'
#_=dream.deepdream(dream.net, img)
#_=googledream.deepdream(googledream.net, img)
##print np.float32(_)
#imsave('itsadream.jpg',_)
caffe.set_mode_cpu()
src = dream.net.blobs['data']
src.data[0] = np.random.randint(0,255,(3,227,227))
jitter=32
ox, oy = np.random.randint(-jitter, jitter+1, 2)
#roll the image
src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)
print 'input', src.data[0]
dream.net.forward()
interested_layers = ['conv1','norm1','pool1','conv2','norm2',
'pool2','conv3','conv4','conv5','pool5','fc6','fc7','fc8','prob']
for i in interested_layers:
    print 'layer data: ',i
    print dream.net.blobs[i].data[0]
    print dream.net.blobs[i].data[0].shape
    print np.sum(dream.net.blobs[i].data[0] > 0)
    print np.sum(dream.net.blobs[i].data[0] == 0)
print "This is how these 2 layers are the same"
print np.sum(dream.net.blobs["fc7"].data[0] == dream.net.blobs["fc8"].data[0])
dst = dream.net.blobs['fc7']
#print 'fc7 data \n', dst.data, dst.data.shape
dst.diff[:] = dst.data
#print 'fc7 gradients \n', dst.diff, dst.diff.shape
mydiff = dream.net.backward(start='fc7',diffs=['fc7','fc6','pool5'])
"""for i in interested_layers[::-1]:
    print 'layer gradient: ',i
    print dream.net.blobs[i].diff[0]
    print dream.net.blobs[i].diff[0].shape
    print np.sum(dream.net.blobs[i].diff[0] > 0)
    print np.sum(dream.net.blobs[i].diff[0] == 0)"""
print mydiff
g = src.diff[0]
print 'derivative',g
