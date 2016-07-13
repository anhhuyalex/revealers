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
#real_img = np.float32(PIL.Image.open('../data/smallPlaces/resizedSmallPlaces/gsun_0a5c663d4ea492022362751c30b6476f.jpg'))
img = np.load('coercedimage1.npy').reshape((3,227,227))
target = np.load('arealimage.npy')

def correlate(x,y):
    xsum = sum(x)
    xavg = xsum/len(x)
    ysum = sum(y)
    yavg = ysum/len(y)
    xdiff = x-xavg
    ydiff = y-yavg
    numerator = xdiff.dot(ydiff)
    xssr = sum((x-xavg)**2.0)
    yssr = sum((y-yavg)**2.0)
    xresiduals = (xssr)**0.5
    yresiduals = (yssr)**0.5
    corr = numerator/xresiduals/yresiduals
    dnumerator = 1./(xresiduals*yresiduals)
    dxdiff = dnumerator*ydiff
    dydiff = dnumerator*xdiff
    dx = np.array(dxdiff)
    dy = np.array(dydiff)
    dxavg = sum(-1.*dxdiff)
    dx += dxavg/len(x)
    dyavg = sum(-1.*dydiff)
    dy += dyavg/len(y)
    ddenom = numerator/((xresiduals*yresiduals)**2.0)
    dxresiduals = yresiduals*ddenom
    dxssr = 0.5/(xssr**0.5)*dxresiduals
    dxsubtraction = 2*(x-xavg)*dxssr
    dx += dxsubtraction
    dx += sum(-1.*dxsubtraction)/len(x)
    dyresiduals = xresiduals*ddenom
    dyssr = 0.5/(yssr**0.5)*dyresiduals
    dysubtraction = 2*(y-yavg)*dyssr
    dy += dysubtraction
    dy += sum(-1.*dysubtraction)/len(y)
    print dx
    return corr,dx,dy

"""caffe.set_mode_cpu()
src = dream.net.blobs['data']
src.data[0] = img
dream.net.forward(end='fc7')
dst = dream.net.blobs['fc7']
x = np.array(dst.data[0])
for i in range(100):
    corr,dx,dy = correlate(x,target)
    print "correlation of this generation: ", corr
    x += dx*500"""

#print 'I am here'
_=dream.deepdream(dream.net, img, target,initial_step_size=1e-7,iter_n=500,decay_rate=1,objective=dream.correlation_gradient,
savefile='coercedimage1')
print _
np.save('coercedimage1.npy',_)
#_=googledream.deepdream(googledream.net, img)
##print np.float32(_)
#imsave('itsadream.jpg',_)
