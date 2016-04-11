'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy as np
import layer_mlp
import layer_conv
import sigmoids
import theanos_MNIST_loader
import readcifar10
import sys, imp
try:
	imp.find_module('matplotlib')
	matplotlibAvailable = True
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	print("matplotlib found, will plot after the last epoch.")
except ImportError:
	matplotlibAvailable = False
	print("Since matplotlib is unavailable, will not plot anything at the end.")

if len(sys.argv) <= 1:
	print("usage:   {mnist|cifar}   {optional: 0 to hide filters/samples}")
	quit()
isCIFAR = False
if len(sys.argv) > 1:
	if 'cifar' in sys.argv[1] or 'Cifar' in sys.argv[1] or 'CIFAR' in sys.argv[1]:
		isCIFAR = True
if isCIFAR:
	print("Will train on CIFAR-10 dataset.")
else:
	print("Will train on MNIST dataset.")

print("removed since currently being submitted as course project UCSB CS 291K")












