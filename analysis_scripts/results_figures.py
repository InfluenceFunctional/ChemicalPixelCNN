import matplotlib.pyplot as plt
import numpy as np
import pickle

# import results
target = 'small_results'
with open(target +'.pkl', 'rb') as f:
    outputs = pickle.load(f)

# plot findings
plt.figure()
# Examples of the training distribution
plt.subplot(2,3,1)
plt.spy(outputs['training_examples'][-1,:,:])
plt.text(outputs['training_examples'].shape[-1]//3, -15, 'Training Example')
plt.text(-15, 5, 'A',fontsize=16,fontweight='bold')
plt.subplot(2,3,4)
plt.spy(outputs['results_examples'][0,:,:])
plt.text(outputs['training_examples'].shape[-1]//3, -15, 'Generated Example')
plt.text(-15, 5, 'D',fontsize=16,fontweight='bold')

# averaging for pair correlation
results2 = np.zeros(len(outputs['results_pair_correlation']))
training2 = np.zeros(len(outputs['training_pair_correlation']))
run = 80
for i in range(len(results2)):
    if i < run:
        results2[i] = np.average(outputs['results_pair_correlation'][0:i])
        training2[i] = np.average(outputs['training_pair_correlation'][0:i])
    else:
        results2[i] = np.average(outputs['results_pair_correlation'][i-run:i])
        training2[i] = np.average(outputs['training_pair_correlation'][i-run:i])

# pair correlation line plot
plt.subplot(2,3,2)
plt.plot(outputs['results_bins'],results2,'.')
plt.plot(outputs['training_bins'],training2,'.')
plt.text(-10, 1.45, 'B',fontsize=16,fontweight='bold')
plt.ylim(0.7,)
plt.xlabel('Radius')
plt.ylabel('Radial Pair Correlation')
plt.legend(['Generated Samples','Training Data'],fontsize=16)

# averaging for radial_frequency
results3 = np.zeros(len(outputs['results_radial_fourier']))
training3 = np.zeros(len(outputs['training_radial_fourier']))
run = 40
for i in range(len(results2)):
    if i < run:
        results3[i] = np.average(outputs['results_radial_fourier'][0:i])
        training3[i] = np.average(outputs['training_radial_fourier'][0:i])
    else:
        results3[i] = np.average(outputs['results_radial_fourier'][i-run:i])
        training3[i] = np.average(outputs['training_radial_fourier'][i-run:i])

# Radial frequency
plt.subplot(2,3,5)
plt.plot(outputs['results_fourier_bins'], results3,'.')
plt.plot(outputs['training_fourier_bins'], training3, '.')
plt.text(-15, 240, 'E',fontsize=16,fontweight='bold')
plt.ylim(5,)
plt.xlabel('Radius')
plt.ylabel('Radial Fourier Signal')

# Local density distribution
plt.subplot(2,3,3)
barwidth = 0.35
plt.bar(np.arange(0,9), outputs['results_interactions_dist'][0], width = barwidth)
plt.bar(np.arange(0,9) + barwidth, outputs['training_interactions_dist'][0], width = barwidth)
plt.xlabel('Particle Neighbors')
plt.ylabel('Population Fraction')
plt.text(-1.75,np.amax(outputs['results_interactions_dist'][0]) * .95, 'C',fontsize=16,fontweight='bold')

# anisotrophy graph
plt.subplot(2,3,6)
plt.imshow(outputs['results_sum']/np.average(outputs['results_sum']),cmap='coolwarm')
cbar = plt.colorbar()
cbar.set_label('Average Density')
plt.text(-15, 5, 'F',fontsize=16,fontweight='bold')