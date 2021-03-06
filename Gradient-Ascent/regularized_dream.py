
# coding: utf-8

# # Deep Dreams (with Caffe)
#
# This notebook demonstrates how to use the [Caffe](http://caffe.berkeleyvision.org/) neural network framework to produce "dream" visuals shown in the [Google Research blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html).
#
# It'll be interesting to see what imagery people are able to generate using the described technique. If you post images to Google+, Facebook, or Twitter, be sure to tag them with **#deepdream** so other researchers can check them out too.
#
# ##Dependencies
# This notebook is designed to have as few dependencies as possible:
# * Standard Python scientific stack: [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [PIL](http://www.pythonware.com/products/pil/), [IPython](http://ipython.org/). Those libraries can also be installed as a part of one of the scientific packages for Python, such as [Anaconda](http://continuum.io/downloads) or [Canopy](https://store.enthought.com/).
# * [Caffe](http://caffe.berkeleyvision.org/) deep learning framework ([installation instructions](http://caffe.berkeleyvision.org/installation.html)).
# * Google [protobuf](https://developers.google.com/protocol-buffers/) library that is used for Caffe model manipulation.

# In[8]:

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
from pylab import norm
from scipy.ndimage.filters import gaussian_filter
import caffe
from scipy.misc import imsave

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

def showarray(a, fmt='jpeg'):
    #uint8: Unsigned integer (0 to 255)
    #Convert image to uint8 TypeError
    #np.clip: values smaller than 0 become 0,
    #values larger than 255 become 255.
    a = np.uint8(np.clip(a, 0, 255))
    #working with text in memory using the file API (read,
    #write. etc.)
    f = StringIO()
    #fromarray: Creates an image memory from an object
    #exporting the array interface (using
    #the buffer protocol).
    #Image.save: Saves this image under the
    #given filename. If no format is
    #specified, the format to use is
    #determined from
    # filename extension, if possible.
    PIL.Image.fromarray(a).save(f, fmt)
    imsave('newimage.jpg',PIL.Image.fromarray(a))
    #display(Image(data=f.getvalue()))


# ## Loading DNN model
# In this notebook we are going to use a [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) model trained on [ImageNet](http://www.image-net.org/) dataset.
# Feel free to experiment with other models from Caffe [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). One particularly interesting [model](http://places.csail.mit.edu/downloadCNN.html) was trained in [MIT Places](http://places.csail.mit.edu/) dataset. It produced many visuals from the [original blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html).

# In[3]:

#model_path = '../../caffe/models/bvlc_googlenet/' # substitute your path here
#net_fn   = model_path + 'deploy.prototxt'
#param_fn = model_path + 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
#Copy Google's Model onto tmp.prototxt
#Initialize a NetParameter Class
#model = caffe.io.caffe_pb2.NetParameter()
#Merge: Parses an ASCII representation of a
#protocol message into a message.
#text_format.Merge(open(net_fn).read(), model)
#model.force_backward = True
#open('tmp.prototxt', 'w').write(str(model))
caffe_root = '/home/ray/caffe/'
model_prototxt = caffe_root + '_temp/deploy.prototxt'
model_trained = caffe_root + 'data/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel'
mean_path = caffe_root + 'features/placesMean.npy'
caffe.set_mode_cpu()
net = caffe.Classifier(model_prototxt, model_trained,
                       mean=np.load(mean_path).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    #reverse it then subtract by the mean
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    #stack arrays along 3rd dimension
    """print np.dstack((img + net.transformer.mean['data'])[::-1]).shape
    print (img + net.transformer.mean['data']).shape"""
    return np.dstack((img + net.transformer.mean['data'])[::-1])
# ##  Producing dreams


# Making the "dream" images is very simple. Essentially it is just a gradient ascent process that tries to maximize the L2 norm of activations of a particular DNN layer. Here are a few simple tricks that we found useful for getting good images:
# * offset image by a random jitter
# * normalize the magnitude of gradient ascent steps
# * apply ascent across multiple scales (octaves)
#
# First we implement a basic gradient ascent step function, applying the first two tricks:

# In[17]:

def objective_L2(dst):
    dst.diff[:] = dst.data

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
    #print dx
    return corr,dx,dy

def correlation_gradient(dst,target):
    corr,grad,dy = correlate(dst.data[0],target)
    print 'features of current image: ', dst.data[0]
    dst.diff[0] = grad
    print 'Correlation of current image with target:', corr

def euclidean_gradient(dst,target):
    dist = (((dst.data[0]-target)**2.0).sum())
    dx = 2*(dst.data[0]-target)

    dst.diff[0] = dx
    print 'Distance of current image features from target:', dist

def both(dst,target):
    corr,grad,dy = correlate(dst.data[0],target)
    print 'features of current image: ', dst.data[0]
    dist = (((dst.data[0]-target)**2.0).sum())
    print 'correlation: ', corr
    print 'measure of similarity: ', dist/100. - corr
    grad *= -1
    grad += 2*(dst.data[0]-target)/100.
    dst.diff[0] = grad

def make_step(net, targetfeatures, v, i, mu = 0.9, step_size=15000, end='inception_4c/output',
              jitter=32, clip=True, ascent=True,objective=euclidean_gradient):
    '''Basic gradient ascent step.'''
    #input image is stored in Net's 'data' blob
    src = net.blobs['data']
    #grab data from layer we want to visualize
    dst = net.blobs[end]
    if i == 0:
        small_val_percentile=50
        print np.percentile(abs(src.data[0]), small_val_percentile)

    net.forward(end=end)
    #print net.blobs[end].data
    # specify the optimization objective
    #set the derivative to be the data from
    #the end layer
    objective(dst,targetfeatures)
    net.backward(start=end)
    #Forwarded only 1 image so we want to take
    #the first
    grad = src.diff[0]
    #print g.shape
    #raise RuntimeError('Stop here')
    v = mu * v - step_size * grad # integrate velocity
    #v = mu * v - step_size * grad # integrate velocity
    # apply normalized ascent step to the input image
    #print v
    #print ascent
    if ascent:
        src.data[0] -= -step_size * grad#/np.abs(g).mean()
        #print step_size/np.abs(g).mean() * g, 'step'
    #gradient descent
    else:
        src.data[0] += -step_size * grad#/np.abs(g).mean()
    #gradient ascent

    #regularize image
    #L2-weight decay
    src.data[0] *= 0.99
    #blur every blur_every gradient ascent steps
    blur_every = 4
    if i % blur_every == 3:
        blur_radius = 1
        for channel in range(3):
            cimg = gaussian_filter(src.data[0,channel], blur_radius)
            src.data[0,channel] = cimg
    #set smallest small_val_percentile% of xx to zero
    small_val_percentile = 5
    small_entries = (abs(src.data[0]) < np.percentile(abs(src.data[0]), small_val_percentile))
    src.data[0] = src.data[0] - src.data[0]*small_entries
    #clip small norms
    small_norm_percentile = 50
    pxnorms = norm(src.data[0], axis=0)
    smallpx = pxnorms < np.percentile(pxnorms, small_norm_percentile)
    smallpx3 = np.tile(smallpx[np.newaxis,:,:], (3,1,1))
    src.data[0] = src.data[0] - src.data[0]*smallpx3
    #clip small contributions
    px_benefit_percentile = 20
    pred_0_benefit = src.diff[0] * -src.data[0]
    px_benefit = pred_0_benefit.sum(1)   # sum over color channels
    smallben = px_benefit < np.percentile(px_benefit, px_benefit_percentile)
    print 'px_benefit shape', px_benefit.shape
    print 'smallben shape', smallben.shape
    smallben3 = np.tile(smallben[np.newaxis,:,:], (3,1,1))
    src.data[0] = src.data[0] - src.data[0]*smallben3
    #clip small absolute benefit
    pred_0_benefit = src.diff[0] * -src.data[0]
    px_benefit = pred_0_benefit.sum(0)   # sum over color channels
    px_abs_benefit_percentile = 10
    smallaben = abs(px_benefit) < np.percentile(abs(px_benefit), px_abs_benefit_percentile)
    smallaben3 = np.tile(smallaben[np.newaxis,:,:], (3,1,1))
    src.data[0] = src.data[0] - src.data[0]*smallaben3
    if clip:
        #bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, 0, 255)
    return v

def deepdream(net, base_img, targetfeatures, iter_n=10, octave_n=4, octave_scale=1.4,
              end='fc7', clip=True, initial_step_size = 15000, decay_rate = 0.6,savefile = 'coercedimage_new',**step_params):
    # prepare base images for all octaves
    #octaves = preprocess(net, base_img)
    #for i in xrange(octave_n-1):
        #octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    print "I'm here"
    src = net.blobs['data']
    src.data[0] = base_img
    #detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    current_step_size = initial_step_size
    v = np.zeros_like(base_img)

    for i in xrange(iter_n):
        print current_step_size
        v = make_step(net, targetfeatures, v, i, end=end, clip=clip, step_size = current_step_size, **step_params)
        current_step_size *= decay_rate
        # visualization
        #vis = deprocess(net, src.data[0])
        #if not clip: # adjust image contrast if clipping is disabled
        #    vis = vis*(255.0/np.percentile(vis, 99.98))
        #showarray(vis)
        #print i, end, vis
        #print deprocess(net, src.data[0])
        #print np.amax(deprocess(net, src.data[0]))
        #np.save('%s.npy'%savefile,deprocess(net, src.data[0]))
        np.save('%s.npy'%savefile,src.data[0])
        #clear_output(wait=True)


    #return deprocess(net, src.data[0])
    return src.data[0]

"""
import numpy as np
from scipy.misc import imsave
imsave('coercedimage.jpg',np.load('coercedimage1.npy'))
for channel in range(3):
    cimg = gaussian_filter(xx[channel], blur_radius)
    xx[channel] = cimg
small_entries = (abs(xx) < percentile(abs(xx), small_val_percentile))
xx = xx - xx*small_entries   # e.g. set smallest 50% of xx to zero
"""
