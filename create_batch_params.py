# generate parameters for batch submission
# data source, layers, f_map, epochs, batch_size, n sampes, sample batches
import numpy as np
import pickle

params ={}
layers = np.array((4, 3, 2, 1)).astype('uint8')
filters = np.array((32, 16)).astype('uint16')
dilation = np.array((1)).astype('uint8')
n_runs= filters.size * layers.size * dilation.size
identity = np.ones(n_runs).astype('uint8')
params['filters'] = [filters[i] for i in range(filters.size) for j in range(layers.size) for k in range(dilation.size)]
params['layers'] = [layers[j] for i in range(filters.size) for j in range(layers.size) for k in range(dilation.size)]
params['dilation'] = [dilation[k] for i in range(filters.size) for j in range(layers.size) for k in range(dilation.size)]
params['model'] = 3 * identity
params['out_maps'] = 2 * identity
params['filter_size'] = 7 * identity
params['bound_type'] = 2 * identity
params['boundary_layers'] = 4 * identity
params['grow_filters'] = 1 * identity
params['training_data'] = 4 * identity
params['training_batch'] = 1024 * identity
params['sample_batch_size'] = 1024 * identity
params['n_samples'] = 2 * identity
params['run_epochs'] = 100 * identity
params['outpaint_ratio'] = 8 * identity
params['noise'] = 0.1 * identity
params['den_var'] = 0.1 * identity
params['GPU'] = 1 * identity
params['TB'] = 1 * identity

with open('batch_parameters.pkl', 'wb') as f:
    pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)