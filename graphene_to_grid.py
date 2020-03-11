import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

'''
This script will 
    1) prepare a sequence of particle positions which are somehow correlated
    2) map these positions to nodes on a 2D grid, with x and y deviations in separate matrices
    3) re-convert from deviation-space to position space 
'''

# generate correlated particle positions (eventually)


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


coords = np.concatenate((np.reshape(np.array(y_coord),(len(y_coord),1)),np.reshape(np.array(x_coord),(len(x_coord),1))),1)

density = len(coords) / np.amax(coords)
edge_size = np.amax(coords) - np.amin(coords)
tot_size = edge_size**2
n_particles = len(coords)
n_samples = 1
coords = np.expand_dims(coords,0)

# map coordinates to a 2D grid
grid_size = int(np.ceil(np.sqrt(n_particles))) # number of gridpoints must equal number of particles
coords = coords * (grid_size - 1) / edge_size # normalize coordinates to grid spacing, -1 keeps proper sizing

grid = np.zeros((n_samples, 2, grid_size, grid_size))
print('Made Coordinates')
coord_list = []
for n in range(n_samples):
    coord_list.append(coords[n,:,:])
check = []
for n in range(n_samples): # this time, search by gridpoint
    grid_rands = np.arange(grid_size ** 2)
    np.random.shuffle(grid_rands)
    for ind in range(len(grid_rands)): # so that we don't get a systemic density fluctuation
        j = (grid_rands[ind] % grid_size)
        i = grid_rands[ind] // grid_size
        if len(coord_list[n] > 0):
            radii = np.sqrt((i-coord_list[n][:,0])**2 + (j-coord_list[n][:,1])**2) # compute all the distances
            particle = np.argmin(radii) # identify index of closest particle
            grid[n, 0, i, j] = i - coord_list[n][particle, 0]  # y-distance
            grid[n, 1, i, j] = j - coord_list[n][particle, 1]  # x-distance
            coord_list[n] = np.delete(coord_list[n],particle,axis=0) # delete elements which we have assigned (speeds up searches)
            # there is a bug in here - it sometimes outputs impossible values - outputs some elements as off by exactly grid_size - indexing error
        check.append(i * grid_size + j)

print('Made grid')
# reconstruct coordinates from 2d grid
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


# discretize grid 256 x 256
maxrange = 5
bins = np.arange(-maxrange, maxrange,2 * maxrange / 256)
image = np.digitize(grid,bins)

print('Finished discretizing')

# re-convert from discretized grid to coords
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

new_coords2 = np.zeros((n_samples,n_particles,2))
for n in range(n_samples):
    for m in range(len(re_coords[n])):
        new_coords2[n,m,:] = re_coords[n][m] # the reconstructed coordinates


# confirm it's a good reconstruction
#x_error = stats.wasserstein_distance(coords[0,:,1],new_coords2[0,:,1])
#y_error = stats.wasserstein_distance(coords[0,:,0],new_coords2[0,:,0])
#total_error = np.sqrt(x_error ** 2 + y_error ** 2)