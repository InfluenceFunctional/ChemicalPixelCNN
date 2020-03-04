import numpy as np
import matplotlib.pyplot as plt

# draw a shape
yrange = 100
xrange = 100
spacing = 10
grad_range = 5 #gradient range


# pixellate
# square grid
#sparse = np.zeros((yrange,xrange))
#for i in range(spacing,yrange):
#    for j in range(spacing,xrange):
#        if (i % spacing == 0) and (j % spacing == 0):
#            sparse[i,j] = 1

#dandom initialization
sparse = np.random.randint(0,spacing**2,size=(yrange,xrange)) == 0

print('Made grid')
5
# fill
gradient = np.zeros((grad_range * 2 + 1, grad_range * 2 + 1)) # build circular gradient
x0 = grad_range # centerpoint
y0 = grad_range
for i in range(grad_range * 2 + 1):
    for j in range(grad_range * 2 + 1):
        gradient[i,j] = 1 - 1/grad_range * np.sqrt(np.abs(y0-i)**2 + np.abs(x0-j)**2)
        if gradient[i,j] < 0:
            gradient[i,j] = 0

filled = np.zeros((yrange,xrange))
for i in range(grad_range,yrange-grad_range):
    for j in range(grad_range,xrange-grad_range):
        if sparse[i,j] == 1:
            filled[i-grad_range:i+grad_range +1, j-grad_range:j+grad_range+1] += gradient # if we find a particle, splat down a gradient around it

print('Filled grid')

# plot
plt.subplot(1,2,1)
plt.imshow(sparse)
plt.subplot(1,2,2)
plt.imshow(filled)