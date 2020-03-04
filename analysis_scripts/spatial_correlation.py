# compute the spatial correlations for a given image
import numpy as np
import matplotlib.pyplot as plt



# load up sample images
#sample = np.load('samples/model=3_dataset=1_filters=16_layers=1_dilation=1_grow_filters=0_filter_size=3_outpaint_ratio=8_epoch=9_generated.npy',allow_pickle=True).astype('uint8')

#sample = np.load('Augmented_Brain_Sample2.npy',allow_pickle=True).astype('uint8')

#sample = np.load('data/repulsive_redo_configs.npy',allow_pickle=True).astype('uint8')
#sample = np.expand_dims(sample,1)

#sample = np.load('data/annealment_redo_configs.npy',allow_pickle=True).astype('uint8')
#sample = np.transpose(sample, [2,1,0])
#sample = np.expand_dims(sample, axis=1)

#sample = np.load('data/sparse_64x64_configs.npy',allow_pickle=True).astype('uint8')
#sample = np.expand_dims(sample,1)

sample = np.load('data/MAC/single_MAC2.npy', allow_pickle=True)
sample = np.expand_dims(sample,1)

max_rad = sample.shape[2]//2 - 1
nbins = max_rad * 10
box_size = 2 * max_rad + 1

# pick a nice central pixel
x0, y0 = [sample.shape[-2]//2 - 1, sample.shape[-1]//2 - 1]

sample = sample[sample[:,:,y0,x0]!=0] # delete samples with zero particles at centroid
sample = sample[:,y0-max_rad:y0+max_rad+1, x0-max_rad:x0+max_rad+1]

# for each pixel within a square box of the appropriate size, assign a radius, coordinates and check its occupancy
corr = np.zeros((box_size, box_size))
corr2 = np.zeros((box_size, box_size))

a, bins = np.histogram(1, bins = nbins, range = (.01, max_rad + 0.01))#max_rad + .01))  # bin the radii
#binned_radii = np.digitize(radius, bins)  # sort for easy retrieval

circle_square_ratio = np.pi/4

dr = bins[1]-bins[0]
N_i = sample.shape[0]  # number of samples
N_tot = np.sum(sample)*circle_square_ratio - N_i # total particle number adjusted for a circular radius and subtracting the centroid
rho = np.average(sample)  # particle density

radial_corr = np.zeros(nbins)
radial_corr2 = np.zeros(nbins)

for i in range(box_size):
    for j in range(box_size):
        if (i != y0) or (j != x0):

            radius= np.sqrt((i - y0) **2 + (j - x0) ** 2)
            corr[i, j] = np.sum(sample[:, i, j]) # density distribution
            corr2[i, j] = corr[i, j] / (radius) # pair-correlation

            if radius <= max_rad: # works better with larger images
                bin_num = np.digitize(radius, bins) - 1
                radial_corr[bin_num] += corr[i, j] #- np.average(sample[:,:,y0,x0],0) * np.average(sample[:,:,iind,jind],0)
                radial_corr2[bin_num] += corr2[i, j] #- np.average(sample[:,:,y0,x0],0) * np.average(sample[:,:,iind,jind],0)

corr[y0,x0] = -np.average(corr)
corr2[y0,x0] = -np.average(corr2)

bin_rad = np.zeros(len(bins)-1)
# kill bins with no) terms
for i in range(len(bins)-1):
    bin_rad[i] = (bins[i] + bins[i+1]) / 2

#bin_rad = bin_rad[radial_corr!=0]
#radial_corr = radial_corr[radial_corr!=0]
#radial_corr2 = radial_corr2[radial_corr2!=0]
radial_corr2 = radial_corr2 / (2 * np.pi * dr * rho * N_i) # this is still not normalized

rolling_mean = np.zeros(len(radial_corr2))
run = 100
for i in range(len(radial_corr2)):
    if i < run:
        rolling_mean[i] = np.average(radial_corr2[0:i])
    else:
        rolling_mean[i] = np.average(radial_corr2[i-run:i])

plt.figure()
plt.subplot(2,2,1)
plt.imshow(corr)
plt.subplot(2,2,2)
plt.plot(bin_rad,radial_corr2 ,'-')
plt.subplot(2,2,3)
plt.imshow(corr2)
plt.subplot(2,2,4)
plt.plot(bin_rad,rolling_mean, '-')