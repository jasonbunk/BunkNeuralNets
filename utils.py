'''Copyright (c) 2015 Jason Bunk
Covered by LICENSE.txt, which contains the "MIT License (Expat)".
'''
import numpy

# e.g. given 50, return a 10x5 shape; or given 81, return 9x9 shape
def MakeNearlySquareImageShape(nUnits):
    imgshapelen = int(round(numpy.sqrt(float(nUnits))))
    if imgshapelen*imgshapelen != nUnits:
        while imgshapelen > 0 and (nUnits % imgshapelen) != 0:
            imgshapelen = imgshapelen - 1
        if imgshapelen > 0:
            return ((nUnits / imgshapelen), imgshapelen)
        else:
            print("WARNING: PlotFiltersSaveImage() had a weird number of visible units: "+str(n_visible))
            return (nUnits, 1)
    else:
        return (imgshapelen, imgshapelen)

# tile images that appear in a matrix, each row of which is an image
# if images are multichannel (RGB), provide a tuple of such matrices each with one of the channels
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(1,1),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
	import theanos_MNIST_loader
	return theanos_MNIST_loader.tile_raster_images(X, img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)

# shuffle entries of a dataset
def shuffle_in_unison(a, b):
	shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
	shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
	permutation = numpy.random.permutation(len(b))
	for old_index, new_index in enumerate(permutation):
		shuffled_a[new_index] = a[old_index]
		shuffled_b[new_index] = b[old_index]
	return (shuffled_a, shuffled_b)
