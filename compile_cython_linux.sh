#!/bin/bash

cython cython_convolution.pyx

gcc -shared -fPIC -O2 -Wall -I/usr/include/python2.7 -o cython_convolution.so cython_convolution.c

