import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from accuracy_metrics import *
from Image_Processing_Utils import *
from spatial_correlation_function import *

# results location
#results_location = 'model=2_dataset=6_filters=16_layers=10_filter_size=7_noise=0.5_denvar=0.1_epoch=9'
#results_location = 'model=2_dataset=7_filters=16_layers=20_filter_size=7_noise=0.5_denvar=0.1_epoch=21'
#results_location = 'model=2_dataset=7_filters=32_layers=20_filter_size=7_noise=0.5_denvar=0.9_epoch=8'
#results_location = 'old_worm'
results_location = 'model=4_dataset=8_filters=32_layers=20_filter_size=7_noise=0.0_denvar=0.0_epoch=6'
with open('outputs/' + results_location + '.pkl', 'rb') as f:
    outputs = pickle.load(f)

result_images = outputs['sample']
result_images = result_images.float()
del outputs

# training set location
# training_images = np.load('drying_sample_1_for_results.npy', allow_pickle=True).astype('bool')
# training_images = np.load('drying_sample_-1.npy', allow_pickle=True).astype('bool')
training_images = np.load('big_worm_results.npy',allow_pickle=True)
training_images = training_images[0:10000,:,:,:]
n_samples = training_images.shape[0]

# density analysis
results_density = np.average(result_images[:,0,:,:],0)
training_density = np.average(training_images[:,0,:,:],0)

# local environment analysis
results_interactions_avg, results_interactions_dist = compute_interactions(result_images)
training_interactions_avg, training_interactions_dist = compute_interactions(torch.Tensor(training_images))

#equalize sizes & samples (if the generated outputs are bigger than the training data)
'''
samples = np.zeros((n_samples // result_images.shape[0] * result_images.shape[0], 1, training_images.shape[-2], training_images.shape[-1]))
for i in range(result_images.shape[0]):
    sample = sample_augment(result_images[i, 0, :, :] == 1, training_images.shape[-2], training_images.shape[-1], 0, 0, 1, n_samples // result_images.shape[0])
    samples[i * (n_samples // result_images.shape[0]): (i + 1) * (n_samples // result_images.shape[0]), 0, :, :] = sample.astype('bool')

result_images = samples
'''

# isotropy
results_sum, results_variance, results_pooled_sum, results_pooled_variance = anisotropy(torch.Tensor(result_images), 0)
training_sum, training_variance, training_pooled_sum, training_pooled_variance = anisotropy(torch.Tensor(training_images), 0)

# fourier analysis
results_fourier_transform = fourier_analysis(result_images)
training_fourier_transform = fourier_analysis(training_images)
results_fourier_bins, results_radial_fourier = radial_fourier_analysis(results_fourier_transform)
training_fourier_bins, training_radial_fourier = radial_fourier_analysis(training_fourier_transform)

# pair correlation
results_radial_density, results_pair_correlation, results_bins = spatial_correlation(result_images)
training_radial_density, training_pair_correlation, training_bins = spatial_correlation(training_images)

# plot findings
plt.figure()
# local dist bar graph
plt.subplot(2,3,1)
barwidth = 0.3
plt.bar(np.arange(0,9), results_interactions_dist[0], width = barwidth)
plt.bar(np.arange(0,9) + barwidth, training_interactions_dist[0], width = barwidth)
plt.xlabel('Particle Neighbors')
plt.ylabel('Population')

# anisotrophy graph
plt.subplot(2,6,7)
plt.imshow(results_sum/np.amax(training_sum))
plt.subplot(2,6,8)
plt.imshow(training_sum/np.amax(training_sum))

# 2D fourier transforms
plt.subplot(2,6,5)
plt.imshow(np.log(np.abs(results_fourier_transform/np.amax(training_fourier_transform))))
plt.subplot(2,6,6)
plt.imshow(np.log(np.abs(training_fourier_transform/np.amax(training_fourier_transform))))

# radial distribution
plt.subplot(2,6,3)
plt.imshow(results_radial_density/np.amax(training_radial_density))
plt.subplot(2,6,4)
plt.imshow(training_radial_density/np.amax(training_radial_density))

# pair correlation line plot
plt.subplot(2,3,5)
plt.plot(results_bins,results_pair_correlation,'.')
plt.plot(training_bins,training_pair_correlation,'.')

# Radial frequency
plt.subplot(2,3,6)
plt.plot(results_fourier_bins, results_radial_fourier,'.')
plt.plot(training_fourier_bins, training_radial_fourier, '.')


outputs = {}
outputs['results_fourier_bins'] = results_fourier_bins
outputs['results_bins'] = results_bins
outputs['results_radial_density'] = results_radial_density
outputs['results_pair_correlation'] = results_pair_correlation
outputs['results_fourier_transform'] = results_fourier_transform
outputs['results_radial_fourier'] = results_radial_fourier
outputs['results_sum'] = results_sum
outputs['results_interactions_dist'] = results_interactions_dist

outputs['training_fourier_bins'] = training_fourier_bins
outputs['training_bins'] = training_bins
outputs['training_radial_density'] = training_radial_density
outputs['training_pair_correlation'] = training_pair_correlation
outputs['training_fourier_transform'] = training_fourier_transform
outputs['training_radial_fourier'] = training_radial_fourier
outputs['training_sum'] = training_sum
outputs['training_interactions_dist'] = training_interactions_dist

outputs['training_examples'] = training_images[0:5,0,:,:]
outputs['results_examples'] = result_images[0:5,0,:,:]


with open('small_results.pkl', 'wb') as f:
    pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)