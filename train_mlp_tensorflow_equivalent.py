"""A very simple MNIST classifer.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
# Import data
import tensorflow as tf
import numpy as np
import cPickle, os, os.path
import theanos_MNIST_loader

sess = tf.InteractiveSession()
printfirststuf = True

def sigma(x):
	return tf.sigmoid(x)

class mlplayer(object):
	def __init__(self, nin, nout, layerName="", saveAndUseSavedWeights=False):
		print("removed since currently being submitted as course project UCSB CS 291K")
	
	#-------------------------------------------------------------------------------
	# Call once, from the first input layer, and returns the network's final output.
	#
	def FeedForwardPredict(self, x):
		print("removed since currently being submitted as course project UCSB CS 291K")

def CostFunction(ypred, ytrue):
	print("removed since currently being submitted as course project UCSB CS 291K")

#---------------------------------------------
print("removed since currently being submitted as course project UCSB CS 291K")

