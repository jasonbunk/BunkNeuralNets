"""A very simple MNIST classifer.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
# Import data
import tensorflow as tf
import numpy as np
import cPickle, os, os.path
import theanos_MNIST_loader

sess = tf.InteractiveSession()
printfirststuf = True

def sigma(x):
	return tf.sigmoid(x)

class mlplayer(object):
	def __init__(self, nin, nout, layerName="", saveAndUseSavedWeights=False):
		self.layerNext = None #if None, this is the end
		self.layerPrev = None #if None, this is the beginning
		wrange = np.sqrt(37.5/float(nin+nout))
		Wval = np.asarray(np.random.uniform(-1.*wrange, wrange, (nin,nout)), dtype=np.float32)
		
		if saveAndUseSavedWeights:
			weightsfname = layerName+"_weights.pkl"
			if os.path.isfile(weightsfname):
				fff = open(weightsfname,'rb')
				Wval = cPickle.load(fff)
				fff.close()
				print("loaded weights of layer \'"+layerName+"\' from file \'"+weightsfname+"\'")
			else:
				fff = open(weightsfname,'wb')
				cPickle.dump(Wval, fff, protocol=cPickle.HIGHEST_PROTOCOL)
				fff.close()
				print("saved weights of layer \'"+layerName+"\' to file \'"+weightsfname+"\'")
		
		self.W = tf.Variable(Wval,layerName+"_W")
		self.b = tf.Variable(tf.zeros([1,nout]),layerName+"_b")
		self.layerName = layerName
		if printfirststuf:
			print(self.layerName+".W.shape == "+str(self.W.get_shape()))
			print(self.layerName+".b.shape == "+str(self.b.get_shape()))
	
	#-------------------------------------------------------------------------------
	# Call once, from the first input layer, and returns the network's final output.
	#
	def FeedForwardPredict(self, x):
		lastz = tf.matmul(x, self.W) + self.b
		aa = sigma(lastz)
		if self.layerNext is None:
			return aa
		else:
			return self.layerNext.FeedForwardPredict(aa)

def CostFunction(ypred, ytrue):
	ydiff = tf.sub(ypred, ytrue)
	return 0.5 * tf.reduce_sum(tf.mul(ydiff, ydiff)) / tf.cast(tf.shape(ydiff)[0], "float")

#---------------------------------------------
Nin = 28*28
Nhidden = 80
Nout = 10
batchsize = 20
maxNumEpochs = 200
lrfixed = 0.07

allDTYPE = np.float32
trainSet, testSet = theanos_MNIST_loader.load_data('data/mnist.pkl.gz')
trainSet = (np.asarray(trainSet[0],dtype=allDTYPE), np.asarray(trainSet[1],dtype=allDTYPE))
testSet  = (np.asarray(testSet[0], dtype=allDTYPE), np.asarray(testSet[1], dtype=allDTYPE))

batchesPerTrainEpoch = int(trainSet[0].shape[0]) / int(batchsize)
batchesPerValidEpoch = int(testSet[0].shape[0]) / int(batchsize)
#---------------------------------------------
# Create the model
layer0 = mlplayer(Nin, Nhidden, "layer1")
layer1 = mlplayer(Nhidden, Nhidden, "layer2")
layer2 = mlplayer(Nhidden, Nout, "layer3")
layer0.layerNext = layer1
layer1.layerNext = layer2
layer2.layerPrev = layer1
layer1.layerPrev = layer0

print("@@@@@@@@@@@@@@@@@ compiling")

xx = tf.placeholder("float", [batchsize,784])
ypre = layer0.FeedForwardPredict(xx)
ytru = tf.placeholder("float", [batchsize,10])

print("@@@@@@@@@@@@@@@@@ training")

# Define loss and optimizer
train_step = tf.train.GradientDescentOptimizer(lrfixed).minimize(CostFunction(ypre,ytru))

correct_prediction = tf.equal(tf.argmax(ypre,1), tf.argmax(ytru,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Train
tf.initialize_all_variables().run()

for epoch in range(maxNumEpochs):
	for batch in xrange(batchesPerTrainEpoch):
		batch_xs = trainSet[0][(batch*batchsize):((batch+1)*batchsize),:]
		batch_ys = trainSet[1][(batch*batchsize):((batch+1)*batchsize),:]
		train_step.run({xx: batch_xs, ytru: batch_ys})
		if batch % 500 == 0:
			loss = CostFunction(ypre,ytru).eval({xx: batch_xs, ytru: batch_ys})
			print("loss at epoch "+str(epoch+1)+"/"+str(maxNumEpochs)+", batch "+str(batch+1)+"/"+str(batchesPerTrainEpoch)+" == "+str(loss))
	validerrtot = 0.
	for batch in xrange(batchesPerValidEpoch):
		batch_xs = testSet[0][(batch*batchsize):((batch+1)*batchsize),:]
		batch_ys = testSet[1][(batch*batchsize):((batch+1)*batchsize),:]
		validerrtot += accuracy.eval({xx: batch_xs, ytru: batch_ys})
	meanvalidacc = validerrtot/float(batchesPerValidEpoch)
	print("@@@@@@@@@@ validation accuracy at epoch "+str(epoch+1)+"/"+str(maxNumEpochs)+" == "+str(meanvalidacc))


