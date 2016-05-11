'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np
import sigmoids
import cPickle, os, os.path

''' MLP layer
=============
This is a "layer", or really a pair of layers - input and output.
It's a set of perceptrons whose outputs form an output layer, and who each take the entire input vector.

With matrix multiplication, given an input row vector x, each column of the weight matrix W is like a single perceptron acting on x.

The set of columns of the matrix W form the set of perceptrons that create the output layer.

Thus the number of rows of W must match the number of input features (dimension of the input vector),
and the number of columns of W is the dimension of the output vector (i.e. the number of parallel perceptions).

MLPlayers are to be used in sequence to form a "multilayer perceptron" (MLP).
Each MLPlayer is a function taking an input vector and outputting another vector.
The input of the first layer is the input of the entire network, and the output of the last layer can be used for classification.

Backpropogation (without batches):
==================================
Imagine a 3-MLPlayer network. Let each MLPlayer be a function f1, f2, f3.
Then the output of the network is f3(f2(f1(x))).

Each MLPlayer consists of two operations, a matrix multiplication and a sigmoidal nonlinearity.
So let each function f be two functions, a matrix multiplication z(x) and a sigmoid s(z), so that f(x) = s(z(x)).
Then the output of the network is s3(z3( s2(z2( s1(z1(x)) )) )).
z(x) will be z(x) = x.W + b, where x is a row vector dotted with each column of W as described above (the dot is the matrix product).

Now define a cost function J(s3) that measures the difference between the produced output and the desired (truth) output;
for example the MSE cost function is J(s3) = 0.5*|s3-y|^2 where y is the truth vector (desired output).

Now the backpropogation rule for weight W1, the lowest MLPlayer, using the chain rule:
dJ/dW1 = dJ/ds3 ds3/dz3 dz3/ds2 ds2/dz2 dz2/ds1 ds1/dz1 dz1/dW1
Where:
dJ/ds3 = s3-y
all dsN/dzN = derivative of sigmoid function with respect to its input = dsN/dzN == ds/dz
dz1/dW1 is basically x since z1(x) = x.W1+b1
analogously all dzN/dsN-1 are basically WN

For upper layers,
dJ/dW2 = dJ/ds3 ds3/dz3 dz3/ds2 ds2/dz2 dz2/dW2  == ld2 dz2/dW2
dJ/dW3 = dJ/ds3 ds3/dz3 dz3/dW3                  == ld3 dz3/dW3

For bias vectors b, dzN/dbN = 1

We have defined ldN == "little delta N" such that
ld3 = dJ/ds3 ds3/dz3 ~~ (s3-y) ds/dz
ld2 = dJ/ds3 ds3/dz3 dz3/ds2 ds2/dz2 == ld3 dz3/ds2 ds2/dz2 ~~ ld3 W3 ds/dz
ld1 = dJ/ds3 ds3/dz3 dz3/ds2 ds2/dz2 dz2/ds1 ds1/dz1 == ld2 dz2/ds1 ds1/dz1 ~~ ld2 W2 ds/dz
where ~~ means "looks something like" but isn't precise.
To be precise (so the matrix math works out), where . is matrix product, * is elementwise multiplication:
ld1 = (ld2 . W2^T) * ds/dz   where ds/dz evaluated at the last produced z1
ld2 = (ld3 . W3^T) * ds/dz   where ds/dz evaluated at the last produced z2
ld3 = (s3-y) * ds/dz
Now:
dJ/dWN = input(Nth MLPlayer)^T . ldN
dJ/dbN = ldN

Backpropogation with batches:
=============================
Input x must be a matrix, each row is a sample input vector.
Output of each MLPlayer will also be a matrix, whose rows are the independent samples.
Little delta ldN will be a matrix with a number of rows equal to the number of input samples; so
dJ/dbN = mean{ldN}
dJ/dWN = mean{matrix multiplication of each row of ldN with corresponding column of (input(Nth MLPlayer)^T)}
'''

class mlplayer(object):
	def __init__(self, nin, nout, sigmoidalfunc=sigmoids.sigmoid, layerName="", saveAndUseSavedWeights=False, checkgradients=False):
		self.layertype = 'mlp'
		self.layerNext = None #if None, this is the end
		self.layerPrev = None #if None, this is the beginning
		self.layerName = layerName
		
		self.checkgradients = checkgradients
		if checkgradients:
			print("Will be checking gradients instead of training the network.")
			DTYPE = np.float64
		else:
			DTYPE = np.float32
		
		self.nin = nin
		self.nout = nout
		wrange = np.sqrt(37.5/float(nin+nout)) #heuristic for the randomly generated weights
		self.W = np.asarray(np.random.uniform(-1.*wrange, wrange, (nin,nout)), dtype=DTYPE)
		self.b = np.zeros((1,nout),dtype=DTYPE) #row vector
		
		if saveAndUseSavedWeights:
			weightsfname = layerName+"_weights.pkl"
			if os.path.isfile(weightsfname):
				fff = open(weightsfname,'rb')
				self.W = cPickle.load(fff)
				fff.close()
				print("loaded weights of layer \'"+layerName+"\' from file \'"+weightsfname+"\'")
			else:
				fff = open(weightsfname,'wb')
				cPickle.dump(self.W, fff, protocol=cPickle.HIGHEST_PROTOCOL)
				fff.close()
				print("saved weights of layer \'"+layerName+"\' to file \'"+weightsfname+"\'")
		
		if checkgradients:
			print(self.layerName+": W.shape == "+str(self.W.shape)+", b.shape == "+str(self.b.shape))
			self.Wshape = self.W.shape
			self.bshape = self.b.shape
		
		self.sigmoidalfunc = sigmoidalfunc
		self.lastinput = None
		self.lastz = None
		self.lasta = None
	
	#-------------------------------------------------------------------------------
	# Call once, from the first input layer, and returns the network's final output.
	#
	def FeedForwardPredict(self, xx):
		self.lastinput = xx
		self.lastz = np.dot(xx, self.W) + self.b
		self.lasta = self.sigmoidalfunc.calc(self.lastz)
		if self.layerNext is None:
			return self.lasta
		else:
			return self.layerNext.FeedForwardPredict(self.lasta)
	
	#-------------------------------------------------------------------------------
	# Call once, from the final output layer, AFTER calling "FeedForwardPredict".
	#
	def BackPropUpdate_MSE(self, yy, learnRate, incomingdelta=None):
		# calculate little delta
		if incomingdelta is None:
			assert(self.layerNext is None)
			assert(yy is not None and self.lasta.shape == yy.shape)
			if yy.dtype != self.W.dtype:
				yy = np.asarray(yy,dtype=self.W.dtype)
			aydiff = np.subtract(self.lasta, yy)
			littledelta = np.multiply(aydiff, self.sigmoidalfunc.calcderiv(self.lastz))
		else:
			assert(self.layerNext is not None)
			littledelta = np.multiply(incomingdelta, self.sigmoidalfunc.calcderiv(self.lastz))
		
		# little delta has a number of rows equal to the number of input samples (and a number of columns equal to the number of outputs)
		db = np.mean(littledelta,axis=0)
		db = np.reshape(db,(1,self.nout))
		
		# dot each row (sample) of littledelta with corresponding row (sample) of the lastinput
		dW = np.zeros(self.W.shape,dtype=self.W.dtype)
		for nn in range(littledelta.shape[0]):
			term1 = np.reshape(littledelta[nn,:], (1,self.nout))
			term2 = np.reshape(self.lastinput[nn,:], (1,self.nin))
			dW += np.dot(np.transpose(term2), term1)
		dW *= (1. / float(littledelta.shape[0]))
		
		if self.checkgradients:
			self.CheckGradient(yy, littledelta, db, dW, self.MSE_Error)
		else:
			# gradient descent; can't update while checking gradients, if checking gradients
			self.b -= db * abs(learnRate)
			self.W -= dW * abs(learnRate)
		
		# propogate backwards; previous layer will not need truth yy except for checking gradients
		if self.layerPrev is not None:
			nextdelta = np.dot(littledelta, np.transpose(self.W))
			if self.layerPrev.layertype == 'conv':
				# reshape nextdelta to the conv layer's output shape
				nextdelta = np.reshape(nextdelta, self.layerPrev.outputshape)
			if self.checkgradients:
				self.layerPrev.BackPropUpdate_MSE(yy, learnRate, nextdelta)
			else:
				self.layerPrev.BackPropUpdate_MSE(None, learnRate, nextdelta)
		
		if self.layerNext is None: #if top layer, return loss (MSE)
			aydiff = np.abs(aydiff)
			return 0.5 * np.sum(np.multiply(aydiff, aydiff)) / float(aydiff.shape[0])
	
	#-------------------------------------------------------------------------------
	# Used for testing.
	#
	def CheckGradient(self, yy, littledelta, db, dW, errorfunc):
		if self.W.dtype != np.float64 or self.b.dtype != np.float64 or littledelta.dtype != np.float64:
			print("gradient checking works best when the numeric type is double precision (np.float64)")
		print(self.layerName+" is checking gradients")
		assert(self.W.shape == self.Wshape)
		assert(self.b.shape == self.bshape)
		assert(littledelta.shape == self.lastz.shape and self.lastz.shape == self.lasta.shape)
		Wsaved = np.copy(self.W)
		bsaved = np.copy(self.b)
		lastinputsaved = np.copy(self.lastinput)
		self.FeedForwardPredict(lastinputsaved)
		origcost = errorfunc(yy)
		
		for ii in range(self.nout):
			self.b = np.copy(bsaved)
			tinyamt = np.abs(self.b[0,ii]) * 0.0001
			if tinyamt < 0.00000000001:
				tinyamt = 0.000001
			self.b[0,ii] += tinyamt
			self.FeedForwardPredict(lastinputsaved)
			newcost = errorfunc(yy)
			newdb = (newcost-origcost)/tinyamt
			denom = 0.5*(np.abs(newdb)+np.abs(db[0,ii]))
			if denom > 0.:
				absdiff = np.abs(newdb-db[0,ii])
				reldiff = absdiff/denom
				if (reldiff > 0.01 and denom > 0.0001) or (absdiff > 0.001 and denom < 3.):
					print("    "+self.layerName+" self.b[0,"+str(ii)+"] == "+str(self.b[0,ii]))
					print("    db[0,"+str(ii)+"] == "+str(db[0,ii])+", calculated dJ/db[0,"+str(ii)+"] == "+str(newdb))
					print("    b[0,"+str(ii)+"]: relative difference == "+str(reldiff)+", denom == "+str(denom))
					print("    origcost == "+str(origcost)+", newcost == "+str(newcost))
					quit()
		self.b = np.copy(bsaved)
		
		for ii in range(self.W.shape[0]):
			for jj in range(self.W.shape[1]):
				self.W = np.copy(Wsaved)
				tinyamt = np.abs(self.W[ii,jj]) * 0.0001
				if tinyamt < 0.00000000001:
					tinyamt = 0.000001
				self.W[ii,jj] += tinyamt
				self.FeedForwardPredict(lastinputsaved)
				newcost = errorfunc(yy)
				newdw = (newcost-origcost)/tinyamt
				denom = 0.5*(np.abs(newdw)+np.abs(dW[ii,jj]))
				if denom > 0.:
					absdiff = np.abs(newdw-dW[ii,jj])
					reldiff = absdiff/denom
					if (reldiff > 0.01 and denom > 0.0001) or (absdiff > 0.001 and denom < 3.):
						print("    "+self.layerName+" self.W["+str(ii)+","+str(jj)+"] == "+str(self.W[ii,jj])+", tinyamt == "+str(tinyamt))
						print("    dW["+str(ii)+","+str(jj)+"] == "+str(dW[ii,jj])+", calculated dJ/dW["+str(ii)+","+str(jj)+"] == "+str(newdw))
						print("    dW["+str(ii)+","+str(jj)+"]: relative difference == "+str(reldiff)+", denom == "+str(denom))
						print("    origcost == "+str(origcost)+", newcost == "+str(newcost))
						quit()
		self.W = np.copy(Wsaved)
	
	#-------------------------------------------------------------------------------
	# Call from the final output layer AFTER calling "FeedForwardPredict".
	#
	def ClassificationAccuracy(self, yy):
		# yy and lasta have shape (nsamples, nclasses)
		# predictions for each sample are argmax amongst classes
		if self.layerNext is None:
			return np.mean(yy[np.arange(yy.shape[0]), np.argmax(self.lasta,axis=1)])
		else:
			return self.layerNext.ClassificationAccuracy(yy)
	
	#-------------------------------------------------------------------------------
	# Call from the final output layer AFTER calling "FeedForwardPredict".
	#
	def MSE_Error(self, yy):
		if self.layerNext is None:
			aydiff = np.abs(np.subtract(self.lasta, yy))
			return 0.5 * np.sum(np.multiply(aydiff, aydiff)) / float(aydiff.shape[0])
		else:
			return self.layerNext.MSE_Error(yy)









