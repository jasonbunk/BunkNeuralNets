'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np

class identity:
	def calc(self,x):
		return x
	def calcderiv(self,x):
		return np.multiply(x,0.)+1.

# ranges from 0 to 1
class sigmoid:
	def calc(self,x):
		return np.divide(1., 1. + np.exp(x*-1.))
	def calcderiv(self,x):
		return np.multiply(self.calc(x), self.calc(x)*-1. + 1.)

# ranges from -1 to 1
class tanh:
	def calc(self,x):
		return np.tanh(x)
	def calcderiv(self,x):
		return 1.0 - np.power(np.tanh(x), 2.)

# ranges from 0 to 1
class tanhadjusted:
	def calc(self,x):
		return (np.tanh(x) + 1.) * 0.5
	def calcderiv(self,x):
		return (1.0 - np.power(np.tanh(x), 2.)) * 0.5

# ranges from 0 to infinity
class relu:
	def calc(self,x):
		return np.maximum(x,0.)
	def calcderiv(self,x):
		return (np.sign(x)+1.)*0.5
