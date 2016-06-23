import numpy as np
import caffe
import sys

# Main path to your caffe installation
caffe_root = '/home/ray/caffe/'

# Model prototxt file
model_prototxt = caffe_root + '_temp/deploy.prototxt'

# Model caffemodel file
model_trained = caffe_root + 'data/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel'

# File containing the class labels
#imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'

# Path to the mean image (used for input processing)
mean_path = caffe_root + 'features/placesMean.npy'

# Name of the layer we want to extract
layer_name = 'conv1'

sys.path.insert(0, caffe_root + 'python')

#size of individual batches
forwardbatchsize = 500

def setupnetwork():
    caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters

    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
    return net

def writefeatures(inputfile,currentbatch,net,forwardbatchsize):

    print 'Reading images from: ', inputfile

    # Setting this to CPU, but feel free to use GPU if you have CUDA installed


    # Loading class labels
    #with open(imagenet_labels) as f:
    #    labels = f.readlines()

    # This prints information about the network layers (names and sizes)
    # You can uncomment this, to have a look inside the network and choose which layer to print
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()
    # print net.blobs['data'].data,"Data blob"
    # Processing one image at a time, printint predictions and writing the vector to a file
    imagefiles = []
    #Append first forwardbatchsize images into imagefiles
    morethan500 = True
    with open(inputfile, 'r') as reader:
        imagecount = 0
        for image_path in reader:
            image_path = image_path.strip()
            #print imagecount
            #print image_path
            input_image = caffe.io.load_image(image_path)
            #print input_image
            imagefiles.append(input_image)
            imagecount += 1
            if imagecount == forwardbatchsize:
                break


    with open(inputfile, 'r') as reader:
        data = reader.read().splitlines(True)
    #Only work if there are more than forwardbatchsize images left
    if len(data) > forwardbatchsize:
        #remove first forwardbatchsize lines of resizedPlacesSample (already processed images)
        #print "All Images appended to imagefiles"
        #print "Removed these imagelines from resizedPlacesSample.txt"
        imagefiles = np.array(imagefiles)
        #reshape image for compatibility with model
        imagefiles = np.reshape(imagefiles,(imagefiles.shape[0],3,227,227))
        #print imagefiles.shape, "Now the real shape"
        print "Forwarding images through network, in batches"
        #initialize variables holding features
        batchconv1 = []
        batchconv3 = []
        batchconv5 = []
        batchfc7 = []
        for i in range(0,forwardbatchsize,10):
            net.forward(data = imagefiles[i:i+10])
            batchconv1.append(net.blobs['conv1'].data)
            batchconv3.append(net.blobs['conv3'].data)
            batchconv5.append(net.blobs['conv5'].data)
            batchfc7.append(net.blobs['fc7'].data)
        print "Done forwarding"
        # print net.blobs[layer_name].data
        #print "Extracting data"
        #save in .npy file
    	np.save('batches/batch%d/places_200000_conv1.npy'%(currentbatch),np.array(batchconv1))
        np.save('batches/batch%d/places_200000_conv3.npy'%(currentbatch),np.array(batchconv3))
        np.save('batches/batch%d/places_200000_conv5.npy'%(currentbatch),np.array(batchconv5))
        np.save('batches/batch%d/places_200000_fc7.npy'%(currentbatch),np.array(batchfc7))
        with open(inputfile, 'w') as removelines:
            removelines.writelines(data[forwardbatchsize:])
        return morethan500
    else:
        print "Less than 500 images remaining"
        print "DIY, please"
        morethan500 = False
        return morethan500


#if __name__ == "__main__":
#    main(sys.argv[1:],currentbatch = 1)
