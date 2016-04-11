'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".

If you have Python-OpenCV installed, you can run this file to peek inside the CIFAR tar and view samples.
'''
import cPickle
from cStringIO import StringIO
import tarfile
import gzip
import os, sys
import numpy as np

def ClassLabelToClassVector(y,nclasses):
    # produce a matrix of shape (nsamples,nclasses)
    # where each row is the truth class index repeated nclasses times e.g. [[1 1 1];[0 0 0];[2 2 2];[1 1 1]...] for 3 classes
    yprocessA = np.dot(np.reshape(y,(y.shape[0],1)), np.ones((1,nclasses)))

    # produce a matrix of shape (nsamples,nclasses)
    # where each row is the range from 0 to nclasses-1, e.g. [[0 1 2];[0 1 2];[0 1 2];[0 1 2]...] for 3 classes
    yprocessB = np.dot(np.ones((y.shape[0],1)), np.reshape(np.arange(nclasses), (1,nclasses)))

    # do the equals operation ("==") element-by-element for the above two matrices
    # for the previously used 3-class example, produces [[0 1 0];[1 0 0];[0 0 1];[0 1 0]...]
    # resultant shape is again (nsamples,nclasses)
    return np.asarray(np.equal(yprocessA, yprocessB), dtype=y.dtype)

def GetDatasetsFromTarfile_DownloadIfNotExist(tarfilepath, normalizeColors=True, convertClassLabelstoClassVectors=False):
    tarfiledir, tarfilename = os.path.split(tarfilepath)
    if tarfiledir == "" and not os.path.isfile(tarfilepath):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0],
                                 "data", tarfilename)
        if os.path.isfile(new_path) or tarfilename == 'cifar-10-python.tar.gz':
            tarfilepath = new_path
    if (not os.path.isfile(tarfilepath)) and tarfilename == 'cifar-10-python.tar.gz':
        import urllib
        origin = ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, tarfilepath)
    return GetDatasetsFromTarfile(tarfilepath, normalizeColors, convertClassLabelstoClassVectors)

def GetDatasetsFromTarfile(tarfilepath, normalizeColors=True, convertClassLabelstoClassVectors=False):
    testdata_x = None
    testdata_y = None
    traindata_x = None
    traindata_y = None
    
    def ReadDict(thedict):
        XX = None
        YY = None
        groupedx = None
        groupedy = None
        grouperIdxCt = 100
        grouperCounter = 0
        numimgs = len(thedict["labels"])
        
        for ii in range(numimgs):
            if groupedx is None:
                groupedx = np.reshape(thedict["data"][ii],(1,3072))
                groupedy = thedict["labels"][ii]
            else:
                groupedx = np.concatenate([groupedx, np.reshape(thedict["data"][ii],(1,3072))])
                groupedy = np.append(groupedy, thedict["labels"][ii])
            
            grouperCounter = (grouperCounter + 1)
            if grouperCounter >= grouperIdxCt:
                if XX is None:
                    XX = groupedx
                    YY = groupedy
                else:
                    XX = np.concatenate([XX, groupedx])
                    YY = np.append(YY, groupedy)
                grouperCounter = 0
                groupedx = None
                groupedy = None
        #print("ReadDict -- XX.shape == "+str(XX.shape)+", YY.shape == "+str(YY.shape))
        return (XX, YY)
    
    tar = tarfile.open(tarfilepath)
    fileinfo = tar.next()
    train_batches = {}
    numtrainbatches = 0
    while fileinfo != None:
        if '_batch' in fileinfo.name:
            #print("loading data from "+fileinfo.name)
            fileobjj = tar.extractfile(fileinfo).read()
            if 'test_batch' in fileinfo.name:
                testdata_x, testdata_y = ReadDict(cPickle.load(StringIO(fileobjj)))
            if 'data_batch' in fileinfo.name:
                numtrainbatches = (numtrainbatches + 1)
                trainbatchnum = int(fileinfo.name[-1])
                train_batches[trainbatchnum] = ReadDict(cPickle.load(StringIO(fileobjj)))
        fileinfo = tar.next()
    
    for trainbatchnum in range(numtrainbatches):
        tbidx = (trainbatchnum + 1) #data batches are indexed 1-5 from 1
        if traindata_x is None:
            traindata_x = train_batches[tbidx][0]
            traindata_y = train_batches[tbidx][1]
        else:
            traindata_x = np.concatenate([traindata_x, train_batches[tbidx][0]])
            traindata_y = np.append(traindata_y, train_batches[tbidx][1])
    
    if normalizeColors:
        divisor = 255.
    else:
        divisor = 1.
    
    #if makeshared:
        #from logistic_sgd import shared_dataset
        #return ( shared_dataset((traindata_x/divisor, traindata_y)), shared_dataset((testdata_x/divisor, testdata_y)) )
    if convertClassLabelstoClassVectors:
        return ((traindata_x/divisor, ClassLabelToClassVector(traindata_y,10)), (testdata_x/divisor, ClassLabelToClassVector(testdata_y,10)))
    else:
        return ((traindata_x/divisor, traindata_y), (testdata_x/divisor, testdata_y))

def LookInside(openedtar):
    import cv2
    fileinfo = openedtar.next()
    while fileinfo != None:
        if '_batch' in fileinfo.name:
            print("opening data file "+fileinfo.name)
            fileobjj = openedtar.extractfile(fileinfo).read()
            if 'test' in fileinfo.name:
                thedict = cPickle.load(StringIO(fileobjj))
                for dictitem in thedict:
                    print("    "+str(dictitem)+" has "+str(len(thedict[dictitem]))+" elements")
                print("        batch_label == "+str(thedict["batch_label"]))
                print(str(thedict["filenames"][:10]))
                print(str(thedict["labels"][:10]))
                for iii in range(50):
                    ''' OpenCV uses BGR format, not RGB format '''
                    BGRimg = np.reshape(thedict["data"][iii][2048:3072], (32,32,1))
                    BGRimg = np.append(BGRimg, np.reshape(thedict["data"][iii][1024:2048], (32,32,1)), axis=2)
                    BGRimg = np.append(BGRimg, np.reshape(thedict["data"][iii][:1024], (32,32,1)), axis=2)
                    cv2.imshow(thedict["filenames"][iii], BGRimg)
                    cv2.waitKey(0)
        fileinfo = openedtar.next()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("usage:  {targzfile}")
        quit()
    tar = tarfile.open(sys.argv[1])
    LookInside(tar)
    tar.close()

