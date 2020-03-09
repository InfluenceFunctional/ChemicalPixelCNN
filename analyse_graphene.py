import numpy as np
import matplotlib.pyplot as plt
from accuracy_metrics import *
import tqdm

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
#nbins = int(maxrange / 0.01)
#pixellate
image, x_bins, y_bins = np.histogram2d(y_coord_small, x_coord_small, nbins)  # pixellate into an image
image = image > 0
'''
# compute fourier
a = np.expand_dims(image.astype(float), 0)
transform = np.average(np.abs(np.fft.fftshift(np.fft.fft2(a))), 0)
fourier_bins, radial_fourier = radial_fourier_analysis(transform)

# compute correlations

#image_set = super_augment(image, 500, 500, 0, 0, 1, 100)
radial_density, pair_correlation, correlation_bin = spatial_correlation(np.expand_dims(np.expand_dims(image,0),0))
'''
#supersample
image = sample_augment(image, 100, 100, 0, 0, 1, 500000).astype('uint8')
image = np.expand_dims(image, 1)

# add a little jitter
for n in tqdm.tqdm(range(len(image))):
    rands = np.random.randint(0, 5, size=600) < 1
    rand = -1

    for i in range(1, image.shape[-2] - 1):
        for j in range(1, image.shape[-1] - 1):
            if image[n,0,i,j] == 1: # if we find a particle, consider moving it a little bit
                rand += 1
                if rands[rand] == 1: # if we go, then sample!
                    new_i = np.random.randint(-1,2)
                    new_j = np.random.randint(-1,2)
                    image[n,0,i,j] = 0
                    image[n,0, i+new_i, j+new_j] = 1

np.save('data/MAC/noisey_graphene.npy', image)