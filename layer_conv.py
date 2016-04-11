'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np
import sigmoids
import cython_convolution

class convlayer(object):
	
	# inshape == (nsamples-per-batch, nchannels-in, im-dimensions...)
	# filtshape == (nchannels-out, nchannels-in, filt-dimensions...)
	# poolshape == (pool-dimensions...) e.g. None or (1,1) for images with no pooling, (2,2) for images with 2x2 pooling kernel
	
	def __init__(self, inshape, filtshape, poolshape=None, sigmoidalfunc=sigmoids.sigmoid, layerName="", checkgradients=False):
		print("removed since currently being submitted as course project UCSB CS 291K")
	
	#-------------------------------------------------------------------------------
	# Call once, from the first input layer, and returns the network's final output.
	#
	def FeedForwardPredict(self, xx):
		print("removed since currently being submitted as course project UCSB CS 291K")
	
	#-------------------------------------------------------------------------------
	# Call once, from the final output layer, AFTER calling "FeedForwardPredict".
	#
	def BackPropUpdate_MSE(self, yy, learnRate, incomingdelta=None):
		print("removed since currently being submitted as course project UCSB CS 291K")
	
	#-------------------------------------------------------------------------------
	# Used for testing.
	#
	def CheckGradient(self, yy, littledelta, db, dW, errorfunc):
		print("removed since currently being submitted as course project UCSB CS 291K")
	
	#-------------------------------------------------------------------------------
	# Call from the final output layer AFTER calling "FeedForwardPredict".
	#
	def MSE_Error(self, yy):
		print("removed since currently being submitted as course project UCSB CS 291K")







