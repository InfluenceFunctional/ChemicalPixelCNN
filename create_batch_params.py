# generate parameters for batch submission
# data source, layers, f_map, epochs, batch_size, n sampes, sample batches
import numpy as np
import pickle

params ={}
layers = np.array((30, 15)).astype('uint8')
filters = np.array((32, 16)).astype('uint16')
dataset = np.array((200000,100000,50000,10000,5000))
n_runs= filters.size * layers.size * dataset.size
identity = np.ones(n_runs).astype('uint8')
params['filters'] = [filters[i] for i in range(filters.size) for j in range(layers.size) for k in range(dataset.size)]
params['layers'] = [layers[j] for i in range(filters.size) for j in range(layers.size) for k in range(dataset.size)]
params['dataset_size'] = [dataset[k] for i in range(filters.size) for j in range(layers.size) for k in range(dataset.size)]
params['model'] = 2 * identity
params['filter_size'] = 7 * identity
params['bound_type'] = 5 * identity
params['boundary_layers'] = 0 * identity
params['training_data'] = 10 * identity
params['training_batch'] = 1024 * identity
params['sample_batch_size'] = 64 * identity
params['softmax_temperature'] = 0.01 * identity
params['n_samples'] = 32 * identity
params['run_epochs'] = 1000 * identity
params['train_margin'] = 1e-6 * identity
params['average_over'] = 5 * identity
params['outpaint_ratio'] = 3 * identity
params['noise'] = 0.2 * identity
params['den_var'] = 0.5 * identity
params['GPU'] = 1 * identity
params['TB'] = 0 * identity

with open('batch_parameters.pkl', 'wb') as f:
    pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)