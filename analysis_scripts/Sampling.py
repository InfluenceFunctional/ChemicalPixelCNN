# start with a single image
# binarize intelligently and subsample / process
from Image_Processing_Utils import *
from os import listdir
from os.path import isfile, join
import torch.nn.functional as F

data_type = 4 # 1 means just from test image (dataset 4), 2 means from our refined set (dataset 5), 3 means from our finite_T dataset

xdim = 128 # sample dimensions
ydim = 128
normalize = 0
binarize = 0
rotflip = 1
pooling_kernel = 1
n_samples = 10000

if data_type == 1:
    image = process_image('data/Test_Image.tif')
    image = F.avgpool2d(correct_brightness(image, 24, 24).unsqueeze(0), pooling_kernel).squeeze(0)
    samples = sample_augment(image, xdim, ydim, normalize, binarize, rotflip, n_samples)
    samples = np.expand_dims(samples, 1).astype('bool')
    np.save('Augmented_Brain_Sample', samples)

elif data_type == 2:
    image_set = []
    path = 'data/Brain_Images'

    images = [f for f in listdir(path) if isfile(join(path,f))]
    for f in images:
        image_set.append(process_image(path + '/' + f))

    samples = np.zeros((n_samples, 1, xdim, ydim))
    for i, image in enumerate(image_set):
        image = F.avg_pool2d(correct_brightness(image, 24, 24).unsqueeze(0), pooling_kernel).squeeze(0)
        sample = sample_augment(image, xdim, ydim, normalize, binarize, rotflip, n_samples // len(image_set))
        samples[i * (n_samples // len(image_set)) : (i+1) * (n_samples // len(image_set)), 0, :, :] = sample.astype('bool')

    np.save('Augmented_Brain_Sample2', samples)

elif data_type == 3: #finite T data
    image_set = np.load('data/256x256_finite_T.npy',allow_pickle=True)[2]

    samples = np.zeros((n_samples, 1, xdim, ydim))
    for i in range(len(image_set)):
        sample = sample_augment(image_set[i][-1,:,:] > 0, xdim, ydim, normalize, binarize, rotflip, n_samples // len(image_set))
        samples[i * (n_samples // len(image_set)): (i + 1) * (n_samples // len(image_set)), 0, :, :] = sample.astype('bool')

    np.save('Finite_T_Sample', samples)

elif data_type == 4: # synthetic drying data
    time_point = 1
    image_set = np.load('C:\OneDrive\McGill/Nanoparticle_Dyn/version6_drying (1)/first_run.npy')

    samples = np.zeros((n_samples, 1, xdim, ydim))
    for i in range(len(image_set)):
        sample = sample_augment(image_set[i][time_point,:,:] == 1, xdim, ydim, normalize, binarize, rotflip, n_samples // len(image_set))
        samples[i * (n_samples // len(image_set)): (i + 1) * (n_samples // len(image_set)), 0, :, :] = sample.astype('bool')

    np.save('drying_sample_%d_for_results'%time_point, samples)
