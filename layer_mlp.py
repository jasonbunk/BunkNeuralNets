'''Copyright (c) 2016 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np
import activations
import costfuncs
import cPickle, os, os.path
#from fastdot import dot as blasdot
def mydotpr(a,b):
	#return blasdot(a,b)
	return np.dot(a,b)

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

Weight decay
=============================
A weight decay term can be added to cost J, this takes the form
lambda*0.5*|Wk|^2 where |Wk| is the Frobenius norm of weight matrix W of layer k.

The total cost function (e.g. for a 3-layer network outputting prediction s3) can then be written as
J(s3) = 0.5*|s3-y|^2 + lambda*0.5*(|W1|^2+|W2|^2+|W3|^2)

Then the gradient dJ/dWk for layer k simply has the additional term lambda*Wk.
Lambda should be set to a small value, perhaps 0.001.

'''

#-------------------------------------------------------------------------------
# Can be used to test prediction accuracy without actually instantiating a layer_mlp.
#
def ClassificationAccuracy(ypred, ytrue):
	return np.mean(ytrue[np.arange(ytrue.shape[0]), np.argmax(ypred,axis=1)])

#-------------------------------------------------------------------------------
# Fully-connected layer of a neural network, described in comments above.
#
class mlplayer(object):
	def __init__(self, nin, nout, activation=activations.sigmoid, costfunc=costfuncs.MSE, L2lambda=0.0, dropoutProb=0.0, useMomentum=False, layerName="", initializationStd=-1, saveAndUseSavedWeights=False, checkgradients=False):
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
		if initializationStd < 0:
			wrange = 6.0*np.sqrt(1.0/float(nin+nout)) #heuristic for the randomly generated weights
			self.W = np.asarray(np.random.uniform(-1.*wrange, wrange, (nin,nout)), dtype=DTYPE)
		else:
			wrange = initializationStd
			self.W = np.asarray(np.random.normal(scale=wrange,size=(nin,nout)), dtype=DTYPE)
		self.b = np.zeros((1,nout),dtype=DTYPE) #row vector
		self.L2lambda = L2lambda
		self.dropoutProb = dropoutProb
		self.useMomentum = useMomentum
		if self.useMomentum:
			self.accumdW = np.zeros(self.W.shape,dtype=self.W.dtype)
			self.accumdb = np.zeros(self.b.shape,dtype=self.b.dtype)
		
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
		
		self.costfunc = costfunc
		self.activation = activation
		self.lastinput = None
		self.lastz = None
		self.lasta = None
	
	#-------------------------------------------------------------------------------
	# Call once, from the first input layer, and returns the network's final output.
	#
	def FeedForwardPredict(self, xx, testTime=False):
		if self.dropoutProb > 0.0 and not testTime:
			dropoutmask = np.random.binomial(n=1,p=(1.0-self.dropoutProb),size=xx.shape)
			self.lastinput = np.multiply(xx,dropoutmask)
		else:
			self.lastinput = xx
		if testTime:
			useweights = (1-self.dropoutProb)*self.W
		else:
			useweights = self.W
		
		self.lastz = mydotpr(self.lastinput, useweights) + self.b
		
		if self.layerNext is None:
			self.lasta = self.lastz # activation not needed on topmost layer
			return self.lasta
		else:
			self.lasta = self.activation.calc(self.lastz)
			return self.layerNext.FeedForwardPredict(self.lasta, testTime=testTime)
	
	#-------------------------------------------------------------------------------
	# Call once, from the final output layer, AFTER calling "FeedForwardPredict".
	#
	def BackPropUpdate(self, yy, learnRate, momentum=0.0, incomingdelta=None):
		# calculate little delta
		if incomingdelta is None or self.layerNext is None:
			assert(self.layerNext is None and incomingdelta is None)
			assert(yy is not None and self.lasta.shape == yy.shape)
			if yy.dtype != self.W.dtype:
				yy = np.asarray(yy,dtype=self.W.dtype)
			#littledelta = np.multiply(self.costfunc.backward(self.lasta, yy), self.activation.calcderiv(self.lastz))
			littledelta = self.costfunc.backward(self.lasta, yy) # activation not needed on topmost layer
		else:
			assert(self.layerNext is not None)
			littledelta = np.multiply(incomingdelta, self.activation.calcderiv(self.lastz))
		
		# little delta has a number of rows equal to the number of input samples (and a number of columns equal to the number of outputs)
		db = np.sum(littledelta,axis=0)
		db = np.reshape(db,(1,self.nout))
		
		# dot each row (sample) of littledelta with corresponding row (sample) of the lastinput
		dW = np.zeros(self.W.shape,dtype=self.W.dtype)
		for nn in range(littledelta.shape[0]):
			term1 = np.reshape(littledelta[nn,:], (1,self.nout))
			term2 = np.reshape(self.lastinput[nn,:], (1,self.nin))
			dW += mydotpr(np.transpose(term2), term1)
		dW += self.L2lambda*self.W
		
		if self.checkgradients:
			if self.layerPrev is not None:
				self.CheckGradient(yy, littledelta, db, dW, self.layerPrev.ComputeCost)
			else:
				self.CheckGradient(yy, littledelta, db, dW, self.ComputeCost)
		else:
			# gradient descent; can't update while checking gradients, if checking gradients
			if self.useMomentum:
				if momentum <= 0.0 or momentum >= 1.0:
					print("warning: momentum outside range (0,1): momentum == "+str(momentum))
				self.accumdb = (self.accumdb * momentum) + db
				self.accumdW = (self.accumdW * momentum) + dW
				self.b -= self.accumdb * abs(learnRate)
				self.W -= self.accumdW * abs(learnRate)
			else:
				self.b -= db * abs(learnRate)
				self.W -= dW * abs(learnRate)
		
		# propogate backwards; previous layer will not need truth yy except for checking gradients
		if self.layerPrev is not None:
			nextdelta = mydotpr(littledelta, np.transpose(self.W))
			if self.layerPrev.layertype == 'conv':
				# reshape nextdelta to the conv layer's output shape
				nextdelta = np.reshape(nextdelta, self.layerPrev.outputshape)
			if self.checkgradients:
				self.layerPrev.BackPropUpdate(yy, learnRate, momentum=momentum, incomingdelta=nextdelta)
			else:
				self.layerPrev.BackPropUpdate(None, learnRate, momentum=momentum, incomingdelta=nextdelta)
	
	#-------------------------------------------------------------------------------
	# Call from the input layer AFTER calling "FeedForwardPredict".
	#
	def ComputeCost(self, yy):
		if self.layerNext is None:
			return self.costfunc.forward(self.lasta, yy) + 0.5*self.L2lambda*np.sum(np.multiply(self.W,self.W))
		else:
			return self.layerNext.ComputeCost(yy) + 0.5*self.L2lambda*np.sum(np.multiply(self.W,self.W))
	
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
		maxeps = 1e-7
		tinyscale  = 0.0001
		maxreldiff = 0.01
		
		for ii in range(self.nout):
			self.b = np.copy(bsaved)
			tinyamt = np.abs(self.b[0,ii]) * tinyscale
			tinyamt = max(tinyamt,maxeps)
			self.b[0,ii] += tinyamt
			self.FeedForwardPredict(lastinputsaved)
			costR = errorfunc(yy)
			self.b = np.copy(bsaved)
			self.b[0,ii] -= tinyamt
			self.FeedForwardPredict(lastinputsaved)
			costL = errorfunc(yy)
			newdb = (costR-costL)/(tinyamt*2.)
			denom = 0.5*(np.abs(newdb)+np.abs(db[0,ii]))
			if denom > 0.:
				absdiff = np.abs(newdb-db[0,ii])
				reldiff = absdiff/denom
				relgrad = denom/max(np.abs(self.b[0,ii]),tinyamt)
				if reldiff > maxreldiff and relgrad > maxreldiff:
					print("    "+self.layerName+" self.b[0,"+str(ii)+"] == "+str(self.b[0,ii]))
					print("    db[0,"+str(ii)+"] == "+str(db[0,ii])+", calculated dJ/db[0,"+str(ii)+"] == "+str(newdb))
					print("    b[0,"+str(ii)+"]: relative difference == "+str(reldiff)+", denom == "+str(denom))
					print("    costL == "+str(costL)+", costR == "+str(costR))
					print("    relgrad == "+str(relgrad))
					quit()
		self.b = np.copy(bsaved)
		
		for ii in range(self.W.shape[0]):
			for jj in range(self.W.shape[1]):
				self.W = np.copy(Wsaved)
				tinyamt = np.abs(self.W[ii,jj]) * tinyscale
				tinyamt = max(tinyamt,maxeps)
				self.W[ii,jj] += tinyamt
				self.FeedForwardPredict(lastinputsaved)
				costR = errorfunc(yy)
				self.W = np.copy(Wsaved)
				self.W[ii,jj] -= tinyamt
				self.FeedForwardPredict(lastinputsaved)
				costL = errorfunc(yy)
				newdw = (costR-costL)/(tinyamt*2.)
				denom = 0.5*(np.abs(newdw)+np.abs(dW[ii,jj]))
				if denom > 0.:
					absdiff = np.abs(newdw-dW[ii,jj])
					reldiff = absdiff/denom
					relgrad = denom/max(np.abs(self.W[ii,jj]),tinyamt)
					if reldiff > maxreldiff and relgrad > maxreldiff:
						print("    "+self.layerName+" self.W["+str(ii)+","+str(jj)+"] == "+str(self.W[ii,jj])+", tinyamt == "+str(tinyamt))
						print("    dW["+str(ii)+","+str(jj)+"] == "+str(dW[ii,jj])+", calculated dJ/dW["+str(ii)+","+str(jj)+"] == "+str(newdw))
						print("    dW["+str(ii)+","+str(jj)+"]: relative difference == "+str(reldiff)+", denom == "+str(denom))
						print("    costL == "+str(costL)+", costR == "+str(costR))
						print("    relgrad == "+str(relgrad))
						quit()
		self.W = np.copy(Wsaved)








