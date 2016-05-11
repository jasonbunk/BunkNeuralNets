'''Copyright (c) 2016 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np


# Classification accuracy is not a "cost function", but it's another useful training metric
def ClassificationAccuracy(ypred, ytrue):
	return np.mean(ytrue[np.arange(ytrue.shape[0]), np.argmax(ypred,axis=1)])


class MSE:
	@staticmethod
	def forward(aa,yy):
		aydiff = np.abs(np.subtract(aa,yy))
		return 0.5*np.sum(np.multiply(aydiff, aydiff)) / float(aa.shape[0])
	@staticmethod
	def backward(aa,yy):
		return np.subtract(aa,yy) / float(aa.shape[0])


class SoftmaxCrossEntropy:	
	@staticmethod
	def probs(aa):
		maxaa = np.amax(aa)
		allexp = np.exp(aa-maxaa)
		allexpsum = np.sum(allexp,axis=1)
		return allexp / np.reshape(allexpsum,(allexpsum.shape[0],1))
	@staticmethod
	def forward(aa,yy):
		softmaxed = SoftmaxCrossEntropy.probs(aa)
		softmaxmult = np.multiply(yy,np.log(softmaxed))
		return -1*np.sum(softmaxmult) / float(aa.shape[0])
	@staticmethod
	def backward(aa,yy):
		softmaxed = SoftmaxCrossEntropy.probs(aa)
		return np.subtract(softmaxed,yy) / float(aa.shape[0])

