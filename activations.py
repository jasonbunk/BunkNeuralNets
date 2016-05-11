'''Copyright (c) 2016 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np

class identity:
	@staticmethod
	def calc(x):
		return x
	@staticmethod
	def calcderiv(x):
		return np.ones(x.shape,dtype=x.dtype)

# ranges from 0 to 1
class sigmoid:
	@staticmethod
	def calc(x):
		return np.divide(1., 1. + np.exp(x*-1.))
	@staticmethod
	def calcderiv(x):
		return np.multiply(sigmoid.calc(x), sigmoid.calc(x)*-1. + 1.)

# ranges from -1 to 1
class tanh:
	@staticmethod
	def calc(x):
		return np.tanh(x)
	@staticmethod
	def calcderiv(x):
		return 1.0 - np.power(np.tanh(x), 2.)

# ranges from 0 to 1
class tanhadjusted:
	@staticmethod
	def calc(x):
		return (np.tanh(x) + 1.) * 0.5
	@staticmethod
	def calcderiv(x):
		return (1.0 - np.power(np.tanh(x), 2.)) * 0.5

# ranges from 0 to infinity
class relu:
	@staticmethod
	def calc(x):
		return np.maximum(x,0.)
	@staticmethod
	def calcderiv(x):
		return (np.sign(x)+1.)*0.5
