'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np
import layer_mlp

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
	Nin = 3*32*32
	trainSet, testSet = readcifar10.GetDatasetsFromTarfile_DownloadIfNotExist('data/cifar-10-python.tar.gz',
							normalizeColors=True, convertClassLabelstoClassVectors=True)
else:
	Nin = 28*28
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

trainSet = (np.asarray(trainSet[0],dtype=allDTYPE), np.asarray(trainSet[1],dtype=allDTYPE))
testSet  = (np.asarray(testSet[0], dtype=allDTYPE), np.asarray(testSet[1], dtype=allDTYPE))

# Datasets should be a pair of matrices; first is input X, second is output Y
# In those matrices X and Y, the rows (0th dimension) index samples.
print("trainSet[0].shape == "+str(trainSet[0].shape))
print("trainSet[1].shape == "+str(trainSet[1].shape))

#---------------------------------------------
Nhidden = 80
Nout = 10
batchsize = 20
maxNumEpochs = 200
batchesPerTrainEpoch = int(trainSet[0].shape[0]) / int(batchsize)
batchesPerValidEpoch = int(testSet[0].shape[0]) / int(batchsize)
LEARNRATE = 0.07
myactivation = activations.sigmoid
mycostfunc = costfuncs.SoftmaxCrossEntropy

#---------------------------------------------
layer0 = layer_mlp.mlplayer(Nin, Nhidden, activation=myactivation, costfunc=mycostfunc, layerName="layer0", checkgradients=CheckNetworkGradients)
layer1 = layer_mlp.mlplayer(Nhidden, Nhidden, activation=myactivation, costfunc=mycostfunc, layerName="layer1", checkgradients=CheckNetworkGradients)
layer2 = layer_mlp.mlplayer(Nhidden, Nout, activation=None, costfunc=mycostfunc, layerName="layer2", checkgradients=CheckNetworkGradients)
layer0.layerNext = layer1
layer1.layerNext = layer2
layer2.layerPrev = layer1
layer1.layerPrev = layer0

firstlayer = layer0
lastlayer = layer2

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
		
		preds = firstlayer.FeedForwardPredict(batch_xs)
		loss = firstlayer.ComputeCost(batch_ys)
		meantrainacc += ClassificationAccuracy(preds, batch_ys)
		lastlayer.BackPropUpdate(batch_ys, LEARNRATE)
		
		numtrainevals += 1
		totaloflosses += loss
		if (batch+1) % 500 == 0:
			print("epoch "+str(epoch+1)+"/"+str(maxNumEpochs)+", batch "+str(batch+1)+"/"+str(batchesPerTrainEpoch)+", loss: "+str(loss)+", training accuracy: "+str((meantrainacc/float(numtrainevals))*100.))
			meantrainacc = 0.
			numtrainevals = 0
	losses.append(totaloflosses / allDTYPE(batchesPerTrainEpoch))
	
	meanvalidacc = 0.
	for batch in range(batchesPerValidEpoch):
		preds = firstlayer.FeedForwardPredict(testSet[0][(batch*batchsize):((batch+1)*batchsize),:])
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
		# each column of the first weight matrix W is a filter over the entire image
		# so transpose layer0.W so that its rows become the filters
		if isCIFAR:
			reshapedW = np.reshape(np.transpose(layer0.W), (layer0.W.shape[1], 3, 1024)) #32*32==1024
			threerows = (reshapedW[:,2,:], reshapedW[:,1,:], reshapedW[:,0,:]) #OpenCV defaults to BGR
			l0filtersimg = utils.tile_raster_images(threerows, (32,32), utils.MakeNearlySquareImageShape(layer0.W.shape[1]))
		else:
			l0filtersimg = utils.tile_raster_images(np.transpose(layer0.W), (28,28), utils.MakeNearlySquareImageShape(layer0.W.shape[1]))
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
	# trainSet = utils.shuffle_in_unison(trainSet[0], trainSet[1])
















