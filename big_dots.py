import numpy as np
import tqdm
import matplotlib.pyplot as plt
from skimage.draw import line_aa, line

# import npy structures
images = np.load('data/MAC/big_perfect_graphene.npy', allow_pickle=True)
#images = np.load('data/MAC/noisey_graphene2.npy', allow_pickle=True)
images = images[:,0,:,:]
#images = np.load('sample1.npy',allow_pickle=True)
#images = np.load('trial_run.npy',allow_pickle=True)
out_image = np.zeros((images.shape[0],images.shape[1],images.shape[2]))

# search each image
for n in tqdm.tqdm(range(images.shape[0])):
    for i in range(images[n, :, :].shape[-2]):
        for j in range(images[n, :, :].shape[-1]):
            if images[n, i, j] != 0:  # if we find a particle
                #out_image[n, i - 2:i + 3, j - 2:j + 3].fill(1) # make it bigger!
                out_image[n, i - 1:i + 2, j - 1:j + 2] += 1  # inner ring


out_image += images
out_image = out_image.astype('uint8')
np.save('data/MAC/big_perfect_graphene', np.expand_dims(out_image,1))
