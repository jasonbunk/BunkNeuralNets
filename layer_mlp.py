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
	def ClassificationAccuracy(self, yy):
		print("removed since currently being submitted as course project UCSB CS 291K")
	
	#-------------------------------------------------------------------------------
	# Call from the final output layer AFTER calling "FeedForwardPredict".
	#
	def MSE_Error(self, yy):
		print("removed since currently being submitted as course project UCSB CS 291K")









