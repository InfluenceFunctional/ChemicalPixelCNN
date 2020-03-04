import numpy as np
import matplotlib.pyplot as plt
from accuracy_metrics import *

# import graphene file
filename = 'data/MAC/graphene3.gro'

x_coord = []
y_coord = []
xyz = open(filename, "r")
xyz.readline()
xyz.readline()

for line in xyz: # read xyz file
    try:
        A = line.split()
        if float(A[-4]) == 0: # if we are flat on the surface
            x_coord.append(float(A[-6]))
            y_coord.append(float(A[-5]))
    except:
        print('end of file')

xyz.close()

#shrink image
maxrange = 30
x_coord_small = []
y_coord_small = []
for i in range(int(len(x_coord))):
    if (x_coord[i] < maxrange) and (y_coord[i] < maxrange):
        x_coord_small.append(x_coord[i])
        y_coord_small.append(y_coord[i])

nbins = int(maxrange / 0.02)
#pixellate
image, x_bins, y_bins = np.histogram2d(y_coord_small, x_coord_small, nbins)  # pixellate into an image
image = image > 0
'''
# compute fourier
a = np.expand_dims(image.astype(float), 0)
transform = np.average(np.abs(np.fft.fftshift(np.fft.fft2(a))), 0)
fourier_bins, radial_fourier = radial_fourier_analysis(transform)

# compute correlations
#supersample

#image_set = super_augment(image, 500, 500, 0, 0, 1, 100)
radial_density, pair_correlation, correlation_bin = spatial_correlation(np.expand_dims(np.expand_dims(image,0),0))
'''