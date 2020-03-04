import numpy as np
from utils import load_all_pickles
import matplotlib.pyplot as plt

path = 'logfiles/Beluga/good_branes/test2/run2'
# unpickle outputs
outputs = load_all_pickles(path)

# find x and y stuff
nruns = len(outputs)
filters = np.zeros(nruns)
epoch = np.zeros(nruns)
filter_size = np.zeros(nruns)
layers = np.zeros(nruns)
n_samples = np.zeros(nruns)

density_overlap = np.zeros(nruns)
interactions_overlap = np.zeros(nruns)
fourier_overlap = np.zeros(nruns)
average_density = np.zeros(nruns)
average_interactions = np.zeros(nruns)
variance = np.zeros(nruns)
pooled_variance = np.zeros(nruns)
samples=[]

for i in range(len(outputs)):
    filters[i] = outputs[i]['filters']
    epoch[i] = outputs[i]['epoch']
    filter_size[i] = outputs[i]['filter size']
    layers[i] = outputs[i]['layers']
    n_samples[i] = outputs[i]['n_samples']

    density_overlap[i] = outputs[i]['density overlap']
    interactions_overlap[i] = outputs[i]['interactions overlap']
    fourier_overlap[i] = outputs[i]['fourier overlap']
    average_density[i] = outputs[i]['average density']
    average_interactions[i] = outputs[i]['average interactions']
    variance[i] = outputs[i]['spatial variance']
    pooled_variance[i] = outputs[i]['pooled variance']
    samples.append(outputs[i]['sample'])

# graph
'''
filters = np.roll(filters[epoch!=-1].reshape(4,6),-1,axis=0)
layers = np.roll(layers[epoch!=-1].reshape(4,6), -1, axis=0)
density_overlap = np.roll(density_overlap[epoch!=-1].reshape(4,6), -1, axis=0)
interactions_overlap = np.roll(interactions_overlap[epoch!=-1].reshape(4,6),-1,axis=0)
fourier_overlap = np.roll(fourier_overlap[epoch!=-1].reshape(4,6), -1, axis = 0)
variance = np.roll(variance[epoch!=-1].reshape(4,6), -1, axis = 0)
pooled_variance = np.roll(pooled_variance[epoch!=-1].reshape(4,6), -1, axis = 0)

filters = np.roll(np.insert(filters[epoch!=-1],0,(0,0,0)).reshape(4,6),-1,axis=0)
layers = np.roll(np.insert(layers[epoch!=-1],0,(0,0,0)).reshape(4,6), -1, axis=0)
density_overlap = np.roll(np.insert(density_overlap[epoch!=-1],0,(0,0,0)).reshape(4,6), -1, axis=0)
interactions_overlap = np.roll(np.insert(interactions_overlap[epoch!=-1],0,(0,0,0)).reshape(4,6),-1,axis=0)
fourier_overlap = np.roll(np.insert(fourier_overlap[epoch!=-1],0,(0,0,0)).reshape(4,6), -1, axis = 0)
variance = np.roll(np.insert(variance[epoch!=-1],0,(0,0,0)).reshape(4,6), -1, axis = 0)
pooled_variance = np.roll(np.insert(pooled_variance[epoch!=-1],0,(0,0,0)).reshape(4,6), -1, axis = 0)
'''

'''
from accuracy_metrics import fourier_analysis
fourier = []
for i in range(len(outputs)):
    fourier.append(fourier_analysis(samples[i][:,:,:,:], 5, 1))

def plot(a):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(samples[a][0, 0, :, :])
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(fourier[a])))
'''