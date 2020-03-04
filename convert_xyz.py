## script to convert .xyz atomic coordinate files to a 2D grid
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from os import listdir
from os.path import isfile, join

nbins_v = 200
grad_range = 5  # gradient range over which carbons will be smeared
grad_bins = 2
#filename = 'data/MAC/step-20041.xyz'
#filename = 'F:\MAC_data\dump_1/step-2449.xsf'
path = '/home/kilgourm/projects/def-simine/MAC_structures/slurm-6019685'

images = []
print('loading all structure files from',path)
for i in range(1,401): # for all the runs
    directory = 'job-6019685_%d'%i
    files = [f for f in listdir(path+'/'+directory) if isfile(join(path+'/'+directory,f))]
    loaded = 0
    iif = -1
    for iif in range(10,100):
        f = 'step-18%d'%iif+'.xsf'
        if (f in files) and (loaded == 0):
            print('loading', f, 'from run %d'%i)
            loaded = 1
            xyz = open(path+'/'+directory+'/'+f, "r")

            #nbins_v = np.linspace(10,1000,50).astype('uint32')
            #error = np.zeros(len(nbins_v))
            #for index in range(1):#len(nbins_v)):

            nbins = nbins_v#[index]
            x_coord = []
            y_coord = []
            xyz.readline()
            xyz.readline()
            xyz.readline()
            xyz.readline()
            xyz.readline()
            xyz.readline()
            n_atoms, aa = xyz.readline().split()
            n_atoms = int(n_atoms)
            for line in xyz: # read xyz file
                try:
                    atom, x, y, z = line.split()
                    x_coord.append(float(x))
                    y_coord.append(float(y))
                except:
                    print('end of file')

            xyz.close()

            image, x_bins, y_bins = np.histogram2d(x_coord,y_coord,nbins) # pixellate into an image
            del x_coord, y_coord
            images.append(image)


    '''
    x_recon = []
    y_recon = []
    for i in range(len(y_bins)-1): # reconstruct the input from the pixellated image
        for j in range(len(x_bins)-1):
            if image[i, j] != 0:
                x_recon.append(np.average((x_bins[i],x_bins[i+1])))
                y_recon.append(np.average((y_bins[j],y_bins[j+1])))
    '''

    # compute the reconstruction error as the wasserstein distance
    #x_error = stats.wasserstein_distance(x_recon,x_coord)
    #y_error = stats.wasserstein_distance(y_recon,y_coord)
    #total_error = np.sqrt(x_error ** 2 + y_error ** 2)
    #error[index] = total_error
    #print('nbins = %d' % nbins, total_error)
    #del x_recon, y_recon, x_coord, y_coord



# build a suitable representation

#smeared dots
'''
gradient = np.zeros((grad_range * 2 + 1, grad_range * 2 + 1)) # build circular gradient

x0 = grad_range # centerpoint
y0 = grad_range
bins = np.linspace(0,grad_range,grad_bins)
for i in range(grad_range * 2 + 1):
    for j in range(grad_range * 2 + 1):
        gradient[i,j] = 1 - 1/grad_range * bins[np.digitize(np.sqrt(np.abs(y0-i)**2 + np.abs(x0-j)**2),bins) - 1]
        if gradient[i,j] < 0:
            gradient[i,j] = 0

yrange = image.shape[-2]
xrange = image.shape[-1]
filled = np.zeros((yrange,xrange))
for i in range(grad_range,yrange-grad_range):
    for j in range(grad_range,xrange-grad_range):
        if image[i,j] == 1:
            filled[i-grad_range:i+grad_range +1, j-grad_range:j+grad_range+1] += gradient # if we find a particle, splat down a gradient around it

print('Filled grid')
'''

#augment the sample
#from Image_Processing_Utils import sample_augment
#sample = sample_augment(image,100,100,0,0,1,100)# * 2 - 1 # augment
sample = images
print('loaded %d'%int(len(images)), 'files')
np.save('1800-step-sample', sample)
#np.save('data/MAC/big_MAC',sample)