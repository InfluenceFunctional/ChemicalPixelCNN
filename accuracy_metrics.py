import numpy as np
import torch
import torch.nn.functional as F
from Image_Processing_Utils import *
from os import listdir
from os.path import isfile, join
import numpy.linalg as linalg

def compute_density(image, GPU): #deprecated
    # return particle density
    density = torch.sum(image, axis=(1,2,3)) / image.shape[2] / image.shape[3]
    average_density = torch.mean(density)
    return density, average_density

def compute_interactions(image): # compute the number of neighbors of a given occupied pixel

    filter_1 = torch.Tensor(((1,1,1),(1,0,1),(1,1,1))).unsqueeze(0).unsqueeze(0) # this filter computes the number of range-1 interactions of each particle
    interactions = np.zeros((image.shape[0], 1, image.shape[2]-2, image.shape[3]-2))
    delta = min(1000, image.shape[0])
    for i in range(image.shape[0]//delta):
        test_image = image[i*1000:(i+1)*1000].type(torch.float32)
        interactions[i*1000:(i+1)*1000, :, :,:] = F.conv2d(test_image, filter_1, padding = 0) * test_image[:,:,1:-1,1:-1] + test_image[:,:,1:-1,1:-1] # number of interactions per occupied pixel + a marker for the occupied pixel, not padded

    # since we will have large and small outputs, we will build the distribution as a sample of individual pixels
    samples = int(1e6)
    r_numbers = [np.random.randint(interactions.shape[0],size=samples), np.random.randint(interactions.shape[2],size=samples), np.random.randint(interactions.shape[3],size=samples)]
    interactions_sample = interactions[r_numbers[0],0,r_numbers[1],r_numbers[2]]
    interactions_hist = np.histogram(interactions_sample, bins=9, range=(0,9), density=True)  # histogram of interaction strength - zero means empty pixel, 1 means occupied with no neighbors - we eliminate the zeros here
    average_interactions = np.average(interactions)  # average number of interactions per occupied particle

    return average_interactions, interactions_hist

def anisotropy(image, GPU):

    distribution = image.sum(0).squeeze(0)  # sum all the images to find anisotropy
    distribution2 = distribution.unsqueeze(0)
    distribution2 = distribution2.unsqueeze(0)

    if GPU == 1:
        distribution = distribution.cpu().detach().numpy()
    else:
        distribution = distribution.detach().numpy()

    variance = np.var(distribution/np.amax(distribution + 1e-5))  # compute the overall variance (ideally small!)

    pooled_distribution = F.max_pool2d(1/(distribution2+1e-5),distribution2.shape[2]//10,distribution2.shape[3]//10)  # pool according to the inverse - extra sensitive to minima - gaps in the distribution
    pooled_distribution = pooled_distribution.squeeze((0))
    pooled_distribution = pooled_distribution.squeeze((0))

    if GPU == 1:
        pooled_distribution = pooled_distribution.cpu().detach().numpy()
    else:
        pooled_distribution = pooled_distribution.detach().numpy()

    pooled_variance = np.var(pooled_distribution/np.amax(pooled_distribution))

    return distribution/np.amax(distribution), variance, pooled_distribution/np.amax(pooled_distribution), pooled_variance

def fourier_analysis(sample):
    sample = sample[:, 0, :, :]
    n_samples = sample.shape[0]
    slice = 10
    batched_image_transform = np.zeros((n_samples // slice, sample.shape[1], sample.shape[2]))
    for i in range(n_samples // slice):
        batched_image_transform[i, :, :] = np.average(np.abs(np.fft.fftshift(np.fft.fft2(sample[i * slice:(i + 1) * slice, :, :]))), 0)  # fourier transform of the original image

    transform = np.average(batched_image_transform, 0)

    return transform

def radial_fourier_analysis(transform): # convert 2D fourier transform to radial coordinates
    x0, y0 = [transform.shape[-2] // 2 , transform.shape[-1] // 2 ]  # pick a nice central pixel
    transform[y0,x0] = 0
    max_rad = transform.shape[-1] // 2 - 1 # maximum radius
    nbins = max_rad * 10  # set number of bins for sorting
    a, bins = np.histogram(1, bins=nbins, range=(.01, max_rad + 0.01))  # bin the possible radii
    radial_fourier = np.zeros(nbins)  # radial density
    radial_fourier2 = np.zeros(nbins)  # radial pair-correlation

    for i in range(transform.shape[-2]):  # for a box of radius max_rad around x0, y0
        for j in range(transform.shape[-1]):
            if (i != y0) or (j != x0):

                radius = np.sqrt((i - y0) ** 2 + (j - x0) ** 2)

                if radius <= max_rad:  # if we are within the circle drawn over the square
                    bin_num = np.digitize(radius, bins) - 1  # add to bin
                    radial_fourier[bin_num] += transform[i, j]
                    radial_fourier2[bin_num] += transform[i, j] / radius

    bin_rad = np.zeros(len(bins) - 1)

    for i in range(len(bins) - 1):
        bin_rad[i] = (bins[i] + bins[i + 1]) / 2  # assign a radius to each bin

    rolling_mean = np.zeros(len(radial_fourier))
    rolling_mean2 = np.zeros(len(radial_fourier2))
    run = 1#int(nbins // 20 * 1)  # length of rolling mean
    for i in range(run, len(radial_fourier2)):
        rolling_mean[i] = np.average(radial_fourier[i - run:i])
        rolling_mean2[i] = np.average(radial_fourier2[i - run:i])

    return bin_rad, rolling_mean2  # normalize the pair-correlation function

def spatial_correlation(image_set):
    xdim = int(image_set.shape[-1]//2)
    ydim = int(image_set.shape[-2]//2)

    density = np.average(image_set)
    n_samples = np.max((len(image_set), 1000)).astype(int) # divide by density so that we actually get the correct number of filled samples

    samples = np.zeros((n_samples//image_set.shape[0]*image_set.shape[0], 1, ydim, xdim))
    for i in range(image_set.shape[0]):
        sample = sample_augment(image_set[i,0,:,:], ydim, xdim, 0, 0, 0, n_samples // image_set.shape[0])
        samples[i * (n_samples // image_set.shape[0]): (i + 1) * (n_samples // image_set.shape[0]), 0, :, :] = sample.astype('bool')

    sample = samples
    del samples

    #preprocess sample
    max_rad = sample.shape[2] // 2 - 1 # the radius to be explored is automatically set to the maximum possible for the sample image
    nbins = max_rad * 10 # set number of bins for sorting
    box_size = 2 * max_rad + 1 # size of box for radial searching
    x0, y0 = [sample.shape[-2]//2 - 1, sample.shape[-1]//2 - 1] # pick a nice central pixel
    sample = sample[sample[:,:,y0,x0]!=0] # delete samples with zero particles at centre (a waste, I know, but you can always just feed it more samples, or get rid of this if you don't need a central particle)
    sample = sample[:,y0-max_rad:y0+max_rad+1, x0-max_rad:x0+max_rad+1] # adjust sample size

    # prep radial bins
    a, bins = np.histogram(1, bins = nbins, range = (.01, max_rad + 0.01)) # bin the possible radii
    circle_square_ratio = np.pi/4  # radio of circle to square area with equal radius

    # prep constants
    dr = bins[1]-bins[0] # differential radius
    N_i = sample.shape[0]  # number of samples
    N_tot = np.sum(sample)*circle_square_ratio - N_i # total particle number adjusted for a circular radius and subtracting the centroid
    rho = np.average(sample)  # particle density

    # initialize outputs
    radial_corr = np.zeros(nbins) # radial density
    radial_corr2 = np.zeros(nbins) # radial pair-correlation
    corr = np.zeros((box_size, box_size)) # 2D density
    corr2 = np.zeros((box_size, box_size)) # 2D pair-correlation

    # for each pixel within a square box of the appropriate size, assign a radius, coordinates and check its occupancy
    for i in range(box_size): # for a box of radius max_rad around x0, y0
        for j in range(box_size):
            if (i != y0) or (j != x0):

                radius= np.sqrt((i - y0) **2 + (j - x0) ** 2)
                corr[i, j] = np.sum(sample[:, i, j]) # density distribution
                corr2[i, j] = corr[i, j] / (radius) # pair-correlation

                if radius <= max_rad: # if we are within the circle drawn over the square
                    bin_num = np.digitize(radius, bins) - 1  # add to bin
                    radial_corr2[bin_num] += corr2[i, j]

    bin_rad = np.zeros(len(bins)-1)

    for i in range(len(bins)-1):
        bin_rad[i] = (bins[i] + bins[i+1]) / 2 #assign a radius to each bin

    radial_corr2 = radial_corr2 / (2 * np.pi * dr * rho * N_i) # normalize the pair-correlation function

    #compute rolling means for correlation functions
    rolling_mean = np.zeros(len(radial_corr2))
    #rolling_mean2 = np.zeros(len(radial_corr))
    run = 1#int(nbins // 20 * 1) # length of rolling mean
    for i in range(run,len(radial_corr2)):
        rolling_mean[i] = np.average(radial_corr2[i-run:i])
        #rolling_mean2[i] = np.average(radial_corr[i-run:i])

    # average out the central points for easier graph viewing
    corr[y0,x0] = np.average(corr)
    #corr2[y0,x0] = np.average(corr2)

    return corr, rolling_mean, bin_rad


def bond_analysis(images, max_range, particle):
    particle = 1
    images = images[:,0,:,:]
    max_bond_length = int(max_range / 0.2)  # avg bond length is about 1.5A, grid is about 0.2 A

    empty = np.zeros((images.shape[0], images.shape[-2] + max_bond_length * 2, images.shape[-1] + max_bond_length * 2))
    empty[:, max_bond_length:-max_bond_length, max_bond_length:-max_bond_length] = images
    images = empty

    #initialize outputs
    bond_order = []
    bond_length = []
    bond_angle = []

    for n in range(images.shape[0]): #search for neighbors
        if len(bond_order) >= 1000:
            break

        for i in range(max_bond_length, images[n, :, :].shape[-2] - max_bond_length):
            for j in range(max_bond_length, images[n, :, :].shape[-1] - max_bond_length):
                if images[n, i, j] == particle:  # if we find a particle
                    radius = []
                    neighborx = []
                    neighbory = []
                    for ii in range(i - max_bond_length, i + max_bond_length + 1):
                        for jj in range(j - max_bond_length, j + max_bond_length + 1):  # search in a ring around it of radius (max bond length)
                            if images[n, ii, jj] == particle:
                                if not ((i == ii) and (j == jj)):  # if we find a particle that is not the original one, store it's location
                                    rad = (np.sqrt((i - ii) ** 2 + (j - jj) ** 2))
                                    if rad <= max_bond_length:
                                        radius.append(rad * 0.2)
                                        neighborx.append(jj)
                                        neighbory.append(ii)

                    bond_order.append(int(len(radius))) # compute bond_order

                    for r in range(int(len(radius))): # compute bond_lengths
                        bond_length.append(radius[r])
                        for q in range(int(len(radius))):# compute bond angles
                            if r != q:
                                v1 = np.array([i,j]) - np.array([neighbory[r], neighborx[r]])
                                v2 = np.array([i,j]) - np.array([neighbory[q], neighborx[q]])
                                c = np.dot(v1,v2) / linalg.norm(v1) / linalg.norm(v2)
                                bond_angle.append(np.arccos(np.clip(c,-1,1)))


    # compute distributions
    bond_order_dist = np.histogram(np.array(bond_order), bins=7, range=(0, 7), density = True)
    bond_length_dist = np.histogram(np.array(bond_length), bins=50, range =(1, 2), density = True)
    bond_angle_dist = np.histogram(np.array(bond_angle), bins=100, range=(0, np.pi), density = True)
    avg_bond_order = np.average(np.array(bond_order))
    avg_bond_length = np.average(np.array(bond_length))
    avg_bond_angle = np.average(np.array(bond_angle))

    return avg_bond_order, bond_order_dist, avg_bond_length, avg_bond_angle, bond_length_dist, bond_angle_dist
    
