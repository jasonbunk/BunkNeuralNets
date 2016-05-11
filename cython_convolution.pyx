#######################################################################################
# Copyright (c) 2015 Jason Bunk
# Covered by LICENSE.txt, which contains the "MIT License (Expat)".
#######################################################################################
import numpy as np
cimport numpy as np
cimport cython
cimport cython.parallel

#fix a datatype for the arrays
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
#DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t

cdef inline unsigned int uint_max(unsigned int a, unsigned int b) nogil: return a if a >= b else b
cdef inline unsigned int uint_min(unsigned int a, unsigned int b) nogil: return a if a <= b else b
cdef inline          int  int_max(         int a,          int b) nogil: return a if a >= b else b
cdef inline          int  int_min(         int a,          int b) nogil: return a if a <= b else b

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)

#######################################################################################
# xshape == (nsamples-per-batch, nchannels-in, im-dimensions...)
# wshape == (nchannels-out, nchannels-in, filt-dimensions...)
# outshape= (nsamples-per-batch, nchannels-out, im-out-dimensions...)
#######################################################################################
# mode 0 == 'valid'
# mode 1 == 'full'
# mode 2 == 'valid-spatial-pooling' where you will want a W with 1 nchannel-in&out that is np.ones()/np.prod(im-dimensions)
#######################################################################################

def my_batch_convolve_images(np.ndarray[DTYPE_t, ndim=4] xbatch, np.ndarray[DTYPE_t, ndim=4] wset, int mode, int poolsizeDim2IfModeIs2=1, int poolsizeDim3IfModeIs2=1):
	
	assert(xbatch.dtype == DTYPE and wset.dtype == DTYPE)
	assert(xbatch.shape[1] == wset.shape[1] or mode == 2)
	
	cdef unsigned int xshape0 = xbatch.shape[0]
	cdef unsigned int xshape1 = xbatch.shape[1]
	cdef int xshape2 = xbatch.shape[2]
	cdef int xshape3 = xbatch.shape[3]
	cdef unsigned int wshape0 = wset.shape[0]
	cdef unsigned int wshape1 = wset.shape[1]
	cdef int wshape2 = wset.shape[2]
	cdef int wshape3 = wset.shape[3]
	cdef unsigned int outshape0 = xshape0
	cdef unsigned int outshape1 = wshape0
	if mode == 2:
		outshape1 = xshape1 # for spatial pooling, nchannels-in == nchannels-out
	
	cdef unsigned int outshape2 = (xshape2 - wshape2 + 1)
	cdef unsigned int outshape3 = (xshape3 - wshape3 + 1)
	if mode == 1: # 'full' convolution
		outshape2 = (xshape2 + wshape2 - 1)
		outshape3 = (xshape3 + wshape3 - 1)
	if mode == 2: # spatial pooling
		outshape2 = xshape2/poolsizeDim2IfModeIs2
		outshape3 = xshape3/poolsizeDim3IfModeIs2
	
	cdef np.ndarray[DTYPE_t, ndim=4] outarr = np.zeros([outshape0,outshape1,outshape2,outshape3], dtype=DTYPE)
	
	cdef unsigned int sidx, widx, chidx
	cdef int wi, wj, wimin, wimax, wjmin, wjmax, ii, jj
	cdef DTYPE_t tempval
	
	if mode == 0:
		for sidx in cython.parallel.prange(outshape0,nogil=True): # loop over samples in batch
			for widx in range(outshape1): # loop over filters in set
				for ii in range(outshape2): # loop over locations in the output array
					for jj in range(outshape3):
						# now that we are at one location in the output array, with one filter...
						# we need to loop over the filter's dimensions (it is 3D)
						tempval = 0
						for chidx in range(wshape1):
							for wi in range(wshape2): # loop over shape of W, the smaller one (and flip W)
								for wj in range(wshape3):
									tempval += wset[widx,chidx,wshape2-1-wi,wshape3-1-wj] * xbatch[sidx,chidx,ii+wi,jj+wj]
						outarr[sidx,widx,ii,jj] += tempval # equivalent to:  = tempval
	elif mode == 2:
		for sidx in cython.parallel.prange(outshape0,nogil=True): # loop over samples in batch
			for chidx in range(outshape1): # loop over nchannels-out == nchannels-in
				for ii in range(outshape2): # loop over locations in the output array
					for jj in range(outshape3):
						# now that we are at one location in the output array, at one channel...
						# we need to loop over the filter's dimensions (it is 2D)
						tempval = 0
						for wi in range(wshape2): # loop over shape of W, the smaller one
							for wj in range(wshape3):
								tempval += wset[0,0,wi,wj] * xbatch[sidx,chidx,(ii*poolsizeDim2IfModeIs2)+wi,(jj*poolsizeDim3IfModeIs2)+wj]
						outarr[sidx,chidx,ii,jj] += tempval # equivalent to:  = tempval
	elif mode == 1:
		if wshape2 >= xshape2 and wshape3 >= xshape3:
			for sidx in cython.parallel.prange(outshape0,nogil=True): # loop over samples in batch
				for widx in range(outshape1): # loop over filters in set
					for ii in range(outshape2): # loop over locations in the output array
						for jj in range(outshape3):
							# now that we are at one location in the output array, with one filter...
							# we need to loop over the filter's dimensions (it is 3D)
							wimin = int_max(0, xshape2-1-ii)
							wjmin = int_max(0, xshape3-1-jj)
							wimax = int_min(xshape2-1, xshape2+wshape2-2-ii) + 1
							wjmax = int_min(xshape3-1, xshape3+wshape3-2-jj) + 1
							tempval = 0
							for chidx in range(wshape1):
								for wi in range(wimin,wimax): # loop over shape of X, the smaller one
									for wj in range(wjmin,wjmax):
										tempval += xbatch[sidx,chidx,xshape2-1-wi,xshape3-1-wj] * wset[widx,chidx,ii+wi+1-xshape2,jj+wj+1-xshape3]
							outarr[sidx,widx,ii,jj] += tempval # equivalent to:  = tempval
		elif wshape2 < xshape2 and wshape3 < xshape3:
			for sidx in cython.parallel.prange(outshape0,nogil=True): # loop over samples in batch
				for widx in range(outshape1): # loop over filters in set
					for ii in range(outshape2): # loop over locations in the output array
						for jj in range(outshape3):
							# now that we are at one location in the output array, with one filter...
							# we need to loop over the filter's dimensions (it is 3D)
							wimin = int_max(0, wshape2-1-ii)
							wjmin = int_max(0, wshape3-1-jj)
							wimax = int_min(wshape2-1, wshape2+xshape2-2-ii) + 1
							wjmax = int_min(wshape3-1, wshape3+xshape3-2-jj) + 1
							tempval = 0
							for chidx in range(wshape1):
								for wi in range(wimin,wimax): # loop over shape of W, the smaller one
									for wj in range(wjmin,wjmax):
										tempval += wset[widx,chidx,wshape2-1-wi,wshape3-1-wj] * xbatch[sidx,chidx,ii+wi+1-wshape2,jj+wj+1-wshape3]
							outarr[sidx,widx,ii,jj] += tempval # equivalent to:  = tempval
		else:
			print("unsupported shape for convolution in my_batch_convolve_images()")
	else:
		print("unsupported convolution mode in my_batch_convolve_images()")
	return outarr

#######################################################################################
# lastinput.shape   == (nsamples-per-batch, nchannels-in, im-in-dimensions...)
# littledelta.shape == (nsamples-per-batch, nchannels-out, im-out-dimensions...)
# dJdW.shape        == (nchannels-out, nchannels-in, filt-dimensions...)
#
# note: filt-dimension = (im-in-dimension) - (im-out-dimension) + 1
#######################################################################################

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)

def my_batch_dJdW(np.ndarray[DTYPE_t, ndim=4] lastinput, np.ndarray[DTYPE_t, ndim=4] littledelta):
	assert(lastinput.shape[2] > littledelta.shape[2] and lastinput.shape[3] > littledelta.shape[3])
	
	cdef unsigned int lastinputshape0 = lastinput.shape[0]
	cdef unsigned int lastinputshape1 = lastinput.shape[1]
	cdef int lastinputshape2 = lastinput.shape[2]
	cdef int lastinputshape3 = lastinput.shape[3]
	cdef unsigned int littledeltashape1 = littledelta.shape[1]
	cdef int littledeltashape2 = littledelta.shape[2]
	cdef int littledeltashape3 = littledelta.shape[3]
	cdef int wshape2 = lastinput.shape[2] - littledeltashape2 + 1
	cdef int wshape3 = lastinput.shape[3] - littledeltashape3 + 1
	
	cdef np.ndarray[DTYPE_t, ndim=4] dJdW = np.zeros([littledeltashape1,lastinputshape1,wshape2,wshape3], dtype=DTYPE)
	cdef DTYPE_t tempval
	
	cdef int sidx, filtidx, chidx
	cdef int wi, wj, ldi, ldj
	
	for sidx in cython.parallel.prange(lastinputshape0,nogil=True): # loop over samples in batch
		for filtidx in range(littledeltashape1): # loop over output filters channels in littledelta
			for chidx in range(lastinputshape1): # loop over input channels in lastinput
				
				#now we do 2D convolution between littledelta[sidx,filtidx,:,:] and flipped(lastinput[sidx,chidx,:,:])
				#then dJdW[filtidx,chidx,:,:] += (that 2d convolution)
				
				for wi in range(wshape2): # loop over the 2D shape of dJdW
					for wj in range(wshape3):
						tempval = 0
						for ldi in range(littledeltashape2): # loop over littledelta (the smaller of the two being convolved)
							for ldj in range(littledeltashape3):
								tempval += littledelta[sidx,filtidx,littledeltashape2-1-ldi,littledeltashape3-1-ldj] * lastinput[sidx,chidx,lastinputshape2-1-wi-ldi,lastinputshape3-1-wj-ldj]
						dJdW[filtidx, chidx, wi, wj] += tempval
	return dJdW













