#######################################################################################
# Copyright (c) 2015 Jason Bunk
# Covered by LICENSE.txt, which contains the "MIT License (Expat)".
#######################################################################################
import numpy as np
cimport numpy as np

#fix a datatype for the arrays
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
#DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t

cdef inline unsigned int uint_max(unsigned int a, unsigned int b): return a if a >= b else b
cdef inline unsigned int uint_min(unsigned int a, unsigned int b): return a if a <= b else b
cdef inline          int  int_max(         int a,          int b): return a if a >= b else b
cdef inline          int  int_min(         int a,          int b): return a if a <= b else b

cimport cython
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
	print("removed since currently being submitted as course project UCSB CS 291K")
	return outarr

#######################################################################################
# lastinput.shape   == (nsamples-per-batch, nchannels-in, im-in-dimensions...)
# littledelta.shape == (nsamples-per-batch, nchannels-out, im-out-dimensions...)
# dJdW.shape        == (nchannels-out, nchannels-in, filt-dimensions...)
#
# note: filt-dimension = (im-in-dimension) - (im-out-dimension) + 1
#######################################################################################

def my_batch_dJdW(np.ndarray[DTYPE_t, ndim=4] lastinput, np.ndarray[DTYPE_t, ndim=4] littledelta):
	print("removed since currently being submitted as course project UCSB CS 291K")
	return dJdW













