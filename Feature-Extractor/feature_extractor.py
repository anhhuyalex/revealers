import numpy as np
import os, sys, getopt

# Main path to your caffe installation
caffe_root = '/home/michelle/caffe/'

# Model prototxt file
model_prototxt = caffe_root + 'models/placesCNN_upgraded/deploy.prototxt'

# Model caffemodel file
model_trained = caffe_root + 'models/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel'

# File containing the class labels
#imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'

# Path to the mean image (used for input processing)
mean_path = caffe_root + 'models/placesCNN_upgraded/placesMean.npy'

# Name of the layer we want to extract
layer_name = ‘fc7’

sys.path.insert(0, caffe_root + 'python')
import caffe

def main(argv):
    inputfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg

    print 'Reading images from "', inputfile
    print 'Writing vectors to "', outputfile

    # Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))

    # Loading class labels
    #with open(imagenet_labels) as f:
    #    labels = f.readlines()

    # This prints information about the network layers (names and sizes)
    # You can uncomment this, to have a look inside the network and choose which layer to print
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()

    # Processing one image at a time, printint predictions and writing the vector to a file
    with open(inputfile, 'r') as reader:
        with open(outputfile, 'w') as writer:
            writer.truncate()
            for image_path in reader:
                image_path = image_path.strip()
                input_image = caffe.io.load_image(image_path)
                #np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.8g')
		np.save(writer, net.blobs[layer_name].data[0])

if __name__ == "__main__":
    main(sys.argv[1:])
