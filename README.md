# BunkNeuralNets
My neural network library for mostly educational purposes - using Python with NumPy and Cython, and is CPU only.

A script for compiling the Cython file on Linux is included.

Cython is needed for multidimensional convolution operations; Python alone would be far too slow.
Cython is not used for multilayer perceptron models as Numpy has all needed operations.

For comparison purposes, there is a file "train_mlp_tensorflow_equivalent.py" which requires TensorFlow to be installed; only that file uses TensorFlow. It can be seen that the TensorFlow implementation of "mlplayer" does not require writing a backpropogation function.

#### Dependences:
python
python-numpy
cython

#### Optional dependencies, needed for visualizations:
python-opencv
python-matplotlib
