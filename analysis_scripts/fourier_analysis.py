import numpy as np
from Image_Processing_Utils import *
from torchvision import utils
import torch

n_test_samples = 1000
#sample = np.load('samples/brain_outpaint_test/16x_outpaint.npy',allow_pickle=True)[:,0,:,:]
sample = torch.Tensor(2,1024,1024)
sample_transform = np.average(np.abs(np.fft.fftshift(np.fft.fft2(sample))),0)

image = process_image('Test_Image.tif') # get the image and cut off black bar at bottom
image = correct_brightness(image, 32, 32)
image = sample_augment(image, sample.shape[1], sample.shape[2], 0, 1, 1, n_test_samples)
#image = image > np.median(image) # binarize with median threshold

slice = 10
batched_image_transform = np.zeros((n_test_samples//slice, sample.shape[1], sample.shape[2]))
for i in range(n_test_samples//slice):
    batched_image_transform[i, :, :] = np.average(np.abs(np.fft.fftshift(np.fft.fft2(image[i*slice:(i+1)*slice,:,:]))), 0) # fourier transform of the original image

image_transform = np.average(batched_image_transform, 0)

tot_overlap = 1 - np.sum(np.abs(image_transform-sample_transform))/np.sum(image_transform)
'''
image_signal = [np.average(image_transform, 0), np.average(image_transform,1)]
sample_signal = [np.average(sample_transform, 0), np.average(sample_transform,1)]

x_overlap = 1 - np.sum(np.abs(image_signal[1] - sample_signal[1])) / np.sum(image_signal[1])
y_overlap = 1 - np.sum(np.abs(image_signal[0] - sample_signal[0])) / np.sum(image_signal[0])
tot_overlap = (x_overlap + y_overlap) / 2
'''
'''
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(np.log(np.abs(image_transform)))
plt.subplot(1,2,2)
plt.imshow(np.log(np.abs(sample_transform)))

plt.figure(2)
plt.subplot(1,2,1)
plt.semilogy(image_signal[0])
plt.semilogy(sample_signal[0])
plt.subplot(1,2,2)
plt.semilogy(image_signal[1])
plt.semilogy(sample_signal[1])
'''
