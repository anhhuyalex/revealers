import extractfeatures
import os,sys,getopt

net = extractfeatures.setupnetwork()
forwardbatchsize = extractfeatures.forwardbatchsize

def main(argv,net,forwardbatchsize):
    inputfile = ''
    outputfile = ''
    """
    getopt.getopt(args, options[, long_options])
    Parses command line options and parameter list.
    args is the argument list to be parsed,
        without the leading reference to the running program
    options is the string of option letters that the script
    long_options, if specified, must be a list of strings
        with the names of the long options which should be supported

    Return value
        return value consists of two elements:
            the first is a list of (option, value) pairs;
            the second is the list of program arguments
                left after the option list was stripped
                (this is a trailing slice of args)
    """
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
    currentbatch = 270
    morethan500 = True
    while morethan500:
        currentbatch += 1
        print "Processing batch: ",currentbatch
        os.mkdir('batches/batch%d' %currentbatch)
        morethan500 = extractfeatures.writefeatures(inputfile,currentbatch,net,forwardbatchsize)




if __name__ == "__main__":
    main(sys.argv[1:],net,forwardbatchsize)
