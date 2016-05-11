'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np
import activations
import costfuncs
import cython_convolution

class convlayer(object):
	
	# inshape == (nsamples-per-batch, nchannels-in, im-dimensions...)
	# filtshape == (nchannels-out, nchannels-in, filt-dimensions...)
	# poolshape == (pool-dimensions...) e.g. None or (1,1) for images with no pooling, (2,2) for images with 2x2 pooling kernel
	
	def __init__(self, inshape, filtshape, poolshape=None, activation=activations.sigmoid, costfunc=costfuncs.MSE, L2lambda=0.0, useMomentum=False, layerName="", checkgradients=False):
		self.layertype = 'conv'
		assert(len(inshape) == len(filtshape))
		assert(len(inshape) >= 3 and len(filtshape) >= 3)
		assert(inshape[1] == filtshape[1])
		if poolshape is not None:
			assert(len(poolshape) == len(inshape[2:]))
		self.layerNext = None #if None, this is the end
		self.layerPrev = None #if None, this is the beginning
		self.layerName = layerName
		
		self.checkgradients = checkgradients
		if checkgradients:
			print("Will be checking gradients instead of training the network. Note: the Cython convolutional code may need to be recompiled for np.float64 precision.")
			DTYPE = np.float64
		else:
			DTYPE = np.float32
		
		self.poolshape = poolshape
		self.inshape = inshape #save these
		self.filtshape = filtshape #save these
		self.outputshape = [inshape[0], filtshape[0]]
		
		# calculate output shape, consider pooling if needed
		for ii in range(len(inshape)-2):
			# the size will be reduced even without pooling due to using 'valid' convolutions (no convolutions outside the boundaries of the input)
			fulloutshape = inshape[2+ii] - filtshape[2+ii] + 1
			if poolshape is not None:
				self.outputshape.append(fulloutshape / poolshape[ii])
				if(round(self.outputshape[-1]) != self.outputshape[-1]):
					print("error: width of output "+str(fulloutshape)+" must be divisible by pooling size "+str(poolshape[ii])+" because here the output shape would be "+str(self.outputshape[-1])+" which has a fractional part")
					quit()
			else:
				self.outputshape.append(fulloutshape)
		
		# initialize filters with random numbers
		wrange = 6.0*np.sqrt(1.0/float(np.prod(inshape[1:])+filtshape[0]*np.prod(filtshape[2:]))) #heuristic for the randomly generated weights
		self.W = np.asarray(np.random.uniform(-1.*wrange, wrange, filtshape), dtype=DTYPE)
		self.b = np.zeros((1,filtshape[0],1,1), dtype=DTYPE) # last dimensions will need to be set correctly if 1D or 3D convolution is to be supported
		if checkgradients:
			print(self.layerName+": W.shape == "+str(self.W.shape)+", b.shape == "+str(self.b.shape))
			print(layerName+".outputshape == "+str(self.outputshape))
			self.Wshape = self.W.shape
			self.bshape = self.b.shape
		
		self.L2lambda = L2lambda
		self.useMomentum = useMomentum
		if self.useMomentum:
			self.accumdW = np.zeros(self.W.shape,dtype=self.W.dtype)
			self.accumdb = np.zeros(self.b.shape,dtype=self.b.dtype)
		
		self.costfunc = costfunc
		self.activation = activation
		self.lastinput = None
		self.lastz = None
		self.lasta = None
	
	#-------------------------------------------------------------------------------
	# Call once, from the first input layer, and returns the network's final output.
	#
	def FeedForwardPredict(self, xx, testTime=False):
		assert(len(xx.shape) >= 3)
		assert(xx.shape[1:] == self.inshape[1:])
		assert(xx.dtype == self.W.dtype)
		self.lastinput = xx
		
		# for each image in a sample batch: convolves the multichannel image with each multichannel filter to produce a multichannel output image
		self.lastz = cython_convolution.my_batch_convolve_images(xx, self.W, 0) + self.b
		
		# reduce output dimensionality using mean pooling, which is like a convolution with ones() with a stride that avoids spatially overlapping the kernels
		if self.poolshape is not None:
			# last dimensions will need to be set correctly if 1D or 3D convolution is to be supported
			meanpoolfilt = np.reshape(np.ones(self.poolshape,dtype=self.W.dtype),(1,1,self.poolshape[0],self.poolshape[1]))
			self.lastz = cython_convolution.my_batch_convolve_images(self.lastz, meanpoolfilt / float(np.prod(self.poolshape)), 2, self.poolshape[0],self.poolshape[1])
		
		# finally, apply sigmoidal nonlinearity (if needed) and feed forward if there is a next layer
		if self.layerNext is None:
			self.lasta = self.lastz # activation not needed on topmost layer
			return self.lasta
		else:
			self.lasta = self.activation.calc(self.lastz)
			if self.layerNext.layertype == 'mlp':
				# flatten outputs
				return self.layerNext.FeedForwardPredict(np.reshape(self.lasta,   (self.lasta.shape[0], np.prod(self.lasta.shape[1:]))   ))
			else:
				return self.layerNext.FeedForwardPredict(self.lasta)
	
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
		
		# upsample if needed
		if self.poolshape is not None:
			notupsampledlildelta = np.copy(littledelta)
			kronkern = np.ones(self.poolshape)
			littledelta = np.zeros((notupsampledlildelta.shape[0], notupsampledlildelta.shape[1], notupsampledlildelta.shape[2]*self.poolshape[0], notupsampledlildelta.shape[3]*self.poolshape[1]), dtype=self.W.dtype)
			scalefact = 1. / float(np.prod(self.poolshape))
			for samp in range(notupsampledlildelta.shape[0]):
				for chan in range(notupsampledlildelta.shape[1]):
					littledelta[samp,chan,:,:] = np.kron(notupsampledlildelta[samp,chan,:,:],kronkern) * scalefact
		
		# transpose the first and second axes of W and flip the rest
		Wflipaxes = [1, 0]
		Wflipaxes.extend(np.arange(len(self.W.shape)-2)+2)
		Wflipme = np.transpose(np.copy(self.W), Wflipaxes)
		if len(self.inshape) == 3:
			db = np.sum(littledelta,axis=(0,2)).reshape(1,littledelta.shape[1],1)
			Wflipped = Wflipme[:,:,::-1] # 1D convolution (e.g. audio)
			print("unsupported convolution dimensions")
		elif len(self.inshape) == 4:
			db = np.sum(littledelta,axis=(0,2,3)).reshape(1,littledelta.shape[1],1,1)
			Wflipped = Wflipme[:,:,::-1,::-1] # 2D convolution (e.g. images)
		elif len(self.inshape) == 5:
			db = np.sum(littledelta,axis=(0,2,3,4)).reshape(1,littledelta.shape[1],1,1,1)
			Wflipped = Wflipme[:,:,::-1,::-1,::-1] # 3D convolution (e.g. video)
			print("unsupported convolution dimensions")
		else:
			print("unsupported convolution dimensions")
		
		# perform a series of 'valid'-sized flat 2D convolutions between channels of littledelta and lastinput
		dW = cython_convolution.my_batch_dJdW(self.lastinput, littledelta)
		dW += self.L2lambda*self.W
		
		if self.checkgradients:
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
		
		assert(self.lastinput.shape == self.inshape)
		
		# propogate backwards; previous layer will not need truth yy except for checking gradients
		if self.layerPrev is not None:
			nextdelta = cython_convolution.my_batch_convolve_images(littledelta, Wflipped, 1)
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
	# Used for testing.
	#
	def CheckGradient(self, yy, littledelta, db, dW, errorfunc):
		if self.W.dtype != np.float64 or self.b.dtype != np.float64 or littledelta.dtype != np.float64:
			print("gradient checking works best when the numeric type is double precision (np.float64)")
		print(self.layerName+" is checking gradients")
		assert (self.W.shape == self.Wshape)
		assert (self.b.shape == self.bshape)
		#assert (littledelta.shape == self.lastz.shape and self.lastz.shape == self.lasta.shape), "littledelta.shape == "+str(littledelta.shape)+", self.lastz.shape == "+str(self.lastz.shape)+", self.lasta.shape == "+str(self.lasta.shape)
		Wsaved = np.copy(self.W)
		bsaved = np.copy(self.b)
		lastinputsaved = np.copy(self.lastinput)
		maxeps = 1e-7
		tinyscale  = 0.0001
		maxreldiff = 0.01
		
		for ii in range(self.b.shape[1]):
			self.b = np.copy(bsaved)
			tinyamt = np.abs(self.b[0,ii]) * tinyscale
			tinyamt = max(tinyamt,maxeps)
			self.b[0,ii,0,0] += tinyamt
			self.FeedForwardPredict(lastinputsaved)
			costR = errorfunc(yy)
			self.b = np.copy(bsaved)
			self.b[0,ii,0,0] -= tinyamt
			self.FeedForwardPredict(lastinputsaved)
			costL = errorfunc(yy)
			newdb = (costR-costL)/(tinyamt*2.)
			denom = 0.5*(np.abs(newdb)+np.abs(db[0,ii]))
			if denom > 0.:
				absdiff = np.abs(newdb-db[0,ii])
				reldiff = absdiff/denom
				relgrad = denom/max(np.abs(self.b[0,ii,0,0]),tinyamt)
				if reldiff > maxreldiff and relgrad > maxreldiff:
					print("    "+self.layerName+" self.b[0,"+str(ii)+",0,0] == "+str(self.b[0,ii,0,0]))
					print("    db[0,"+str(ii)+"] == "+str(db[0,ii])+", calculated dJ/db[0,"+str(ii)+"] == "+str(newdb))
					print("    b[0,"+str(ii)+"]: relative difference == "+str(reldiff)+", denom == "+str(denom))
					print("    costL == "+str(costL)+", costR == "+str(costR))
					print("    relgrad == "+str(relgrad))
					quit()
		self.b = np.copy(bsaved)
		print(self.layerName+" done checking db; checking dW")
		for ii in range(self.W.shape[0]):
			print(self.layerName+" so far, on ii == "+str(ii)+"/"+str(self.W.shape[0]))
			for jj in range(self.W.shape[1]):
				for kk in range(self.W.shape[2]):
					for ll in range(self.W.shape[3]):
						self.W = np.copy(Wsaved)
						tinyamt = np.abs(self.W[ii,jj,kk,ll]) * tinyscale
						tinyamt = max(tinyamt,maxeps)
						self.W[ii,jj,kk,ll] += tinyamt
						self.FeedForwardPredict(lastinputsaved)
						costR = errorfunc(yy)
						self.W = np.copy(Wsaved)
						self.W[ii,jj,kk,ll] -= tinyamt
						self.FeedForwardPredict(lastinputsaved)
						costL = errorfunc(yy)
						newdw = (costR-costL)/(tinyamt*2.)
						denom = 0.5*(np.abs(newdw)+np.abs(dW[ii,jj,kk,ll]))
						if denom > 0.:
							absdiff = np.abs(newdw-dW[ii,jj,kk,ll])
							reldiff = absdiff/denom
							relgrad = denom/max(np.abs(dW[ii,jj,kk,ll]),tinyamt)
							if reldiff > maxreldiff and relgrad > maxreldiff:
								print("    "+self.layerName+" self.W["+str(ii)+","+str(jj)+","+str(kk)+","+str(ll)+"] == "+str(self.W[ii,jj,kk,ll])+", tinyamt == "+str(tinyamt))
								print("    dW["+str(ii)+","+str(jj)+","+str(kk)+","+str(ll)+"] == "+str(dW[ii,jj,kk,ll])+", calculated dJ/dW["+str(ii)+","+str(jj)+","+str(kk)+","+str(ll)+"] == "+str(newdw))
								print("    dW["+str(ii)+","+str(jj)+","+str(kk)+","+str(ll)+"]: relative difference == "+str(reldiff)+", denom == "+str(denom))
								print("    costL == "+str(costL)+", costR == "+str(costR))
								print("    relgrad == "+str(relgrad))
								quit()
		self.W = np.copy(Wsaved)
























