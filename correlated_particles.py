import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from grid_utils import *
import tqdm

'''
generate a set of particle positions with some simple correlation, maybe waves!
'''

n_samples = 100
xdim = 100
ydim = 100
density = 10
n_particles = int(xdim*ydim*density)
grids = []
for n in tqdm.tqdm(range(n_samples)):
    init = np.random.rand(n_particles, 2) * xdim # initialize scatter
    del_inds = np.zeros(0)
    for ind in range(len(init)): # pare off unwanted particles
        x0 = init[ind,1]
        y0 = init[ind,0]
        if (int(y0 + x0) % 2) == 0:
            del_inds = np.append(del_inds,ind)

    init = np.delete(init, del_inds.astype(int),axis=0)

    grids.append(build_deviation_grid(init,1))

sizes = np.zeros((n_samples,2))
for n in range(n_samples):
    sizes[n,:] = [grids[n].shape[-2], grids[n].shape[-1]]

ymin = np.amin(sizes).astype(int)
xmin = ymin
sample = np.zeros((n_samples,2,ymin,xmin))
for n in range(n_samples):
    sample[n,:,:,:] = grids[n][0,:,0:ymin,0:xmin]

sample = sample[:,:,ymin//4:-ymin//4, xmin//4:-xmin//4] # toss half the data to get a cleaner picture with less artifacts
disc = discretize_grid(sample, 2, 64) # discretize grid distortions over a certain range

coords2 = discrete_to_xyz(disc, 2, 64)