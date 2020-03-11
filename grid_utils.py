import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

'''
This file contains functions for 
    1) load a sequence of particle positions
    2) map these positions to nodes on a 2D grid, with x and y deviations in separate matrices
    3) re-convert from deviation-space to position space 
'''

def import_coordinates(filename): # load particle positions
    x_coord = []
    y_coord = []
    xyz = open(filename, "r")
    xyz.readline()
    xyz.readline()
    for line in xyz: # read xyz file - update as necessary for different file formats
        try:
            A = line.split()
            if float(A[-4]) == 0: # if we are flat on the surface
                x_coord.append(float(A[-6]))
                y_coord.append(float(A[-5]))
        except:
            print('end of file')

    xyz.close()
    print('Made Coordinates')
    coords = np.concatenate((np.reshape(np.array(y_coord),(len(y_coord),1)),np.reshape(np.array(x_coord),(len(x_coord),1))),1)

    return coords


def build_deviation_grid(coords, n_samples):
    density = len(coords) / np.amax(coords) # estimate particle density
    edge_size = np.amax(coords) - np.amin(coords) # estimate size of square area covered by coords
    tot_size = edge_size**2  # count pixels
    n_particles = len(coords) # count number of particles

    coords = np.expand_dims(coords,0)

    # map coordinates to a 2D grid
    grid_size = int(np.ceil(np.sqrt(n_particles))) # number of gridpoints should be about equal to or greater than number of particles
    coords = coords * (grid_size - 1) / edge_size # normalize coordinates to grid spacing, -1 keeps proper sizing

    '''
    There are other ways of doing this which may work better. This approach results in some gridpoints
    being assigned large deviations, because of local density fluctuations (all nearby gridspoints may
    be used up before all the particles in a certain area are assigned). This could be improved by 
    a seconary search which reorganizes to minimize deviations, OR using grid with more points than 
    we have atoms to assign. This latter approach is probably most elegant, however we would have to
    change the search algorithm to find the closes gridpoint for each atom, rather than the closest
    atom for each gridpoint, which results in larger than necessary deviations.
    '''

    grid = np.zeros((n_samples, 2, grid_size, grid_size)) # the grid we will output
    coord_list = []
    for n in range(n_samples):
        coord_list.append(coords[n,:,:])

    for n in range(n_samples): # assign a particle to each gridpoint
        grid_rands = np.arange(grid_size ** 2)
        np.random.shuffle(grid_rands)
        for ind in range(len(grid_rands)): # choose gridpoints stochastically to avoid large systematic error
            j = (grid_rands[ind] % grid_size) # convert 1D list to 2D coords
            i = grid_rands[ind] // grid_size
            if len(coord_list[n] > 0):
                radii = np.sqrt((i-coord_list[n][:,0])**2 + (j-coord_list[n][:,1])**2) # compute all the distances
                particle = np.argmin(radii) # identify index of closest particle
                grid[n, 0, i, j] = i - coord_list[n][particle, 0]  # y-distance
                grid[n, 1, i, j] = j - coord_list[n][particle, 1]  # x-distance
                coord_list[n] = np.delete(coord_list[n],particle,axis=0) # delete elements which we have assigned (speeds up searches)

    print('Made grid')

    # reconstruct coordinates from 2d grid to confirm the grid was properly made
    re_coords = []
    for n in range(n_samples):
        re_coords.append([])

    for n in range(n_samples):
        for i in range(grid.shape[-2]):
            for j in range(grid.shape[-1]):
                if grid[n,0,i,j] != 0:
                    re_coords[n].append((i - grid[n,0,i,j], j - grid[n,1,i,j]))

    new_coords = np.zeros((n_samples,n_particles,2))
    for n in range(n_samples):
        for m in range(len(re_coords[n])):
            new_coords[n,m,:] = re_coords[n][m] # the reconstructed coordinates

    # confirm it's a good reconstruction
    x_error = stats.wasserstein_distance(coords[0,:,1],new_coords[0,:,1])
    y_error = stats.wasserstein_distance(coords[0,:,0],new_coords[0,:,0])
    total_error = np.sqrt(x_error ** 2 + y_error ** 2)

    if total_error > 0.01:
        print('Large Reconstruction Error!')

    return grid

def discretize_grid(grid, maxrange, nbins): # discretize grid distortions over a certain range
    bins = np.arange(-maxrange, maxrange, 2 * maxrange / nbins)
    image = np.digitize(grid,bins)

    print('Finished discretizing')
    return image

def discrete_to_xyz(image, maxrange, nbins): # convert discretized grid back to coordinate space
    # re-convert from discretized grid to coords
    bins = np.arange(-maxrange, maxrange, 2 * maxrange / nbins)
    n_samples = image.shape[0]
    grid_in = image
    delta = bins[1]-bins[0]
    re_coords = []
    for n in range(n_samples):
        re_coords.append([])

    for n in range(n_samples):
        for i in range(grid_in.shape[-2]):
            for j in range(grid_in.shape[-1]):
                if grid_in[n,0,i,j] != 0:
                    re_coords[n].append((i - grid_in[n,0,i,j] * delta + maxrange, j - grid_in[n,1,i,j] * delta + maxrange))

    new_coords2 = np.zeros((n_samples,image.shape[-1] * image.shape[-2],2)) # will be some blanks
    for n in range(n_samples):
        for m in range(len(re_coords[n])):
            new_coords2[n,m,:] = re_coords[n][m] # the reconstructed coordinates

    return new_coords2