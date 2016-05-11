'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np
import layer_mlp
import layer_conv
import activations
import costfuncs
from costfuncs import ClassificationAccuracy
import theanos_MNIST_loader
import readcifar10
import sys, imp
import utils
try:
	imp.find_module('matplotlib')
	matplotlibAvailable = True
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	print("matplotlib found, will plot after the last epoch.")
except ImportError:
	matplotlibAvailable = False
	print("Since matplotlib is unavailable, will not plot anything at the end.")

if len(sys.argv) <= 1:
	print("usage:   {mnist|cifar}   {optional: 0 to hide filters/samples}")
	quit()
isCIFAR = False
if len(sys.argv) > 1:
	if 'cifar' in sys.argv[1] or 'Cifar' in sys.argv[1] or 'CIFAR' in sys.argv[1]:
		isCIFAR = True
if isCIFAR:
	print("Will train on CIFAR-10 dataset.")
else:
	print("Will train on MNIST dataset.")

CheckNetworkGradients = False # this prevents training
ViewFilters = True
ViewSamples = True
if len(sys.argv) > 2:
	try:
		if int(sys.argv[2]) == 0:
			ViewFilters = False
			ViewSamples = False
			print("will NOT be displaying filters or samples")
	except:
		ViewFilters = True

if ViewFilters or ViewSamples:
	print("note: viewing samples/filters requires python-opencv")
	import cv2
if isCIFAR:
	oneimshape = (3,32,32)
	trainSet, testSet = readcifar10.GetDatasetsFromTarfile_DownloadIfNotExist('data/cifar-10-python.tar.gz',
							normalizeColors=True, convertClassLabelstoClassVectors=True)
else:
	oneimshape = (1,28,28)
	trainSet, testSet = theanos_MNIST_loader.load_data('data/mnist.pkl.gz')

if CheckNetworkGradients:
	allDTYPE = np.float64
else:
	allDTYPE = np.float32

# This can make the dataset significantly smaller so the training process appears faster.
# The end result will not have as good of a generalization performance.
if False:
	trainSet = (trainSet[0][:8000,:], trainSet[1][:8000,:])
	testSet = (testSet[0][:4000,:], testSet[1][:4000,:])

trainSet = (np.reshape(trainSet[0], (trainSet[0].shape[0], oneimshape[0], oneimshape[1], oneimshape[2])), trainSet[1])
testSet  = (np.reshape(testSet[0],  (testSet[0].shape[0],  oneimshape[0], oneimshape[1], oneimshape[2])), testSet[1])

trainSet = (np.asarray(trainSet[0],dtype=allDTYPE), np.asarray(trainSet[1],dtype=allDTYPE))
testSet  = (np.asarray(testSet[0], dtype=allDTYPE), np.asarray(testSet[1], dtype=allDTYPE))

# Datasets should be a pair of arrays; first is input X, second is output Y
# In those arrays X and Y, the rows (0th dimension) index samples.
print("trainSet[0].shape == "+str(trainSet[0].shape))
print("trainSet[1].shape == "+str(trainSet[1].shape))

#---------------------------------------------
if CheckNetworkGradients:
	filtsizes = (5, 5)
	nfilters = (4, 6)
	poolsizes = (2, 2)
	nhiddens = (80, 10)
	#filtsizes = (13, 10)
	#nfilters = (11, 12)
	#poolsizes = (1, 1)
	#nhiddens = (10,)
else:
	filtsizes = (5, 5)
	nfilters = (20, 50)
	poolsizes = (2, 2)
	nhiddens = (500, 10)

batchsize = 20
maxNumEpochs = 200
batchesPerTrainEpoch = int(trainSet[0].shape[0]) / int(batchsize)
batchesPerValidEpoch = int(testSet[0].shape[0]) / int(batchsize)
LEARNRATE = 0.01
myactivation = activations.sigmoid
mycostfunc = costfuncs.SoftmaxCrossEntropy

#---------------------------------------------
nextimshape = (batchsize, oneimshape[0], oneimshape[1], oneimshape[2])
print("-1 imshape == "+str(nextimshape))
netlayers = []

#build convolutional layers
for idx in range(len(filtsizes)):
	filtshape = (nfilters[idx], nextimshape[1], filtsizes[idx], filtsizes[idx])
	mypoolshape = (poolsizes[idx], poolsizes[idx])
	netlayers.append(layer_conv.convlayer(nextimshape, filtshape, poolshape=mypoolshape, activation=myactivation, layerName="convlayer"+str(len(netlayers)), checkgradients=CheckNetworkGradients))
	if len(netlayers) > 1:
		netlayers[-1].layerPrev = netlayers[-2]
	nextimshape = (batchsize, nfilters[idx], netlayers[-1].outputshape[2], netlayers[-1].outputshape[3])
	print(str(idx)+" imshape == "+str(nextimshape))

#build fully-connected layers
nextnumin = np.prod(nextimshape[1:])
for idx in range(len(nhiddens)):
	netlayers.append(layer_mlp.mlplayer(nextnumin, nhiddens[idx], activation=myactivation, costfunc=mycostfunc, layerName="fclayer"+str(len(netlayers)), checkgradients=CheckNetworkGradients))
	if len(netlayers) > 1:
		netlayers[-1].layerPrev = netlayers[-2]
	nextnumin = nhiddens[idx]

#build forward links (to next layers)
for idx in range(len(netlayers)-1):
	netlayers[idx].layerNext = netlayers[idx+1]

#---------------------------------------------

epochindicessaved = []
valaccuracies = []
losses = []

if matplotlibAvailable and (ViewFilters or ViewSamples):
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

for epoch in range(maxNumEpochs):
	totaloflosses = 0.
	meantrainacc = 0.
	numtrainevals = 0
	for batch in range(batchesPerTrainEpoch):
		batch_xs = trainSet[0][(batch*batchsize):((batch+1)*batchsize),:]
		batch_ys = trainSet[1][(batch*batchsize):((batch+1)*batchsize),:]
		
		preds = netlayers[0].FeedForwardPredict(batch_xs)
		loss = netlayers[0].ComputeCost(batch_ys)
		meantrainacc += ClassificationAccuracy(preds, batch_ys)
		netlayers[-1].BackPropUpdate(batch_ys, LEARNRATE)
		
		numtrainevals += 1
		totaloflosses += loss
		if (batch+1) % 50 == 0:
			print("epoch "+str(epoch+1)+"/"+str(maxNumEpochs)+", batch "+str(batch+1)+"/"+str(batchesPerTrainEpoch)+", loss: "+str(loss)+", training accuracy: "+str((meantrainacc/float(numtrainevals))*100.))
			meantrainacc = 0.
			numtrainevals = 0
	losses.append(totaloflosses / allDTYPE(batchesPerTrainEpoch))
	
	meanvalidacc = 0.
	for batch in range(batchesPerValidEpoch):
		preds = netlayers[0].FeedForwardPredict(testSet[0][(batch*batchsize):((batch+1)*batchsize),:])
		meanvalidacc += ClassificationAccuracy(preds, testSet[1][(batch*batchsize):((batch+1)*batchsize),:])
	meanvalidacc /= allDTYPE(batchesPerValidEpoch)
	print("@@@@@@@@@@ validation accuracy at epoch "+str(epoch+1)+"/"+str(maxNumEpochs)+" == "+str(meanvalidacc))
	epochindicessaved.append(allDTYPE(epoch+1))
	valaccuracies.append(meanvalidacc)
	
	if ViewSamples:
		selectedsamples = testSet[0][np.random.choice(testSet[0].shape[0], 100, replace=False),:]
		if isCIFAR:
			reshapeds = np.reshape(selectedsamples, (100, 3, 1024)) #32*32==1024
			threerows = (reshapeds[:,2,:], reshapeds[:,1,:], reshapeds[:,0,:]) #OpenCV defaults to BGR
			tiledsamples = utils.tile_raster_images(threerows, (32,32), (10,10))
		else:
			tiledsamples = utils.tile_raster_images(selectedsamples, (28,28), (10,10))
		cv2.imshow("tiledsamples", tiledsamples)
		cv2.waitKey(100)
	
	if ViewFilters:
		n0filters = netlayers[0].W.shape[0]
		filt0size = netlayers[0].W.shape[2]*netlayers[0].W.shape[3]
		filt0shape = (netlayers[0].W.shape[2],netlayers[0].W.shape[3])
		if isCIFAR:
			reshapedW = np.reshape(netlayers[0].W, (n0filters, 3, filt0size))
			threerows = (reshapedW[:,2,:], reshapedW[:,1,:], reshapedW[:,0,:]) #OpenCV defaults to BGR
			l0filtersimg = utils.tile_raster_images(threerows, filt0shape, utils.MakeNearlySquareImageShape(n0filters))
		else:
			l0filtersimg = utils.tile_raster_images(np.reshape(netlayers[0].W,(n0filters,filt0size)), filt0shape, utils.MakeNearlySquareImageShape(n0filters))
		cv2.imshow("l0filters", l0filtersimg)
		cv2.waitKey(100)
	
	if matplotlibAvailable and (ViewFilters or ViewSamples):
		ax1.cla()
		ax2.cla()
		ax1.plot(epochindicessaved, valaccuracies, 'b-')
		ax1.set_xlabel('epoch')
		# Make the y-axis label and tick labels match the line color.
		ax1.set_ylabel('validation accuracy', color='b')
		for tl in ax1.get_yticklabels():
			tl.set_color('b')
		ax2.plot(epochindicessaved, losses, 'r-')
		ax2.set_ylabel('loss (value of cost function)', color='r')
		for tl in ax2.get_yticklabels():
			tl.set_color('r')
		plt.show(False)
		plt.draw()
	
	# shuffling the dataset before the next epoch improves training
	trainSet = utils.shuffle_in_unison(trainSet[0], trainSet[1])
















