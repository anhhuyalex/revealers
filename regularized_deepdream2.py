import numpy as np
#import googledream
import regularized_dream
#real_img = np.float32(PIL.Image.open('../data/smallPlaces/resizedSmallPlaces/gsun_0a5c663d4ea492022362751c30b6476f.jpg'))
img = np.load('regularized_coerced_image.npy').reshape((3,227,227))
#img = np.random.randint(0,255,(3,227,227))
target = np.load('arealimage.npy')

#print 'I am here'
_=regularized_dream.deepdream(regularized_dream.net, img, target,clip=True,initial_step_size=1e7,
iter_n=500,decay_rate=0.97,ascent=False, objective=regularized_dream.both,
savefile='regularized_coerced_image')
print _
np.save('regularized_coerced_image.npy',_)
#_=googledream.deepdream(googledream.net, img)
##print np.float32(_)
#imsave('itsadream.jpg',_)
