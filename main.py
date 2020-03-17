from torch import nn, optim, cuda, backends
from utils import *
from models import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

run = get_input()
## parameters
if run == -1:  # user-defined parameters
    # architecture
    model = 2 # 2 is PixelCNN with pre-activated residual bottlenecks, #DEPRECATED 1 is no-residual PixelCNN, 3 is PixelDRN, 4 - DensePixelCNN
    filters = 5 # number of convolutional filters or feature maps (at initial layer for model ==3)
    filter_size = 7  # initial layer convolution size
    layers = 2 # number of hidden convolutional layers in models 1 & 2, number of layers per block (6 blocks + stem) in model 3
    bound_type = 5  # type of boundary layer, 0 = empty, 1 = seed in top left only, 2 = seed + random noise with appropriate density, 4 = multiple seeds, 5 = large graphene seed, only works for sample batch <= 64 currently #DEPRECATED 3 = generated bound
    boundary_layers = 0 # number of layers of conv_fields between sample and boundary
    softmax_temperature = .01 # ratio to batch mean at which softmax will sample

    # training
    training_data = 10 # select training set: 1 - repulsive, 2 - annealed, 3 - 64x64 finite T, 4 - single brane, 5- refined branes, 6 - synthetic drying, age =1, 7 - synthetic drying, age = -1, 8- small sample synthetic drying age = -1, 9-MAC test image, 10 - graphene
    training_batch = 256 * (2 ** 1) # siz`e of training and test batches - it will try to run at this size, bu5t if it doesn't fit it will go smaller
    sample_batch_size = 16 # batch size for sample generator - will auto-adjust to be sm/eq than training_batch
    n_samples = sample_batch_size//2  # total samples to be generated when we generate, must not be zero (it may make more if there is available memory)
    run_epochs = 1000 # number of incremental epochs which will be trained over - if zero, will run just the generator
    dataset_size = 100  # the maximum number of samples to consider from our dataset
    train_margin = 1e-2  # the convergence criteria for training error
    average_over = 5 # how many epochs to average over to determine convergence
    outpaint_ratio = 0.25 # sqrt of size of output relative to input
    noise = .2 # mean of border noise
    den_var = 0.5 # standard deviation of density variation, also toggle 0 to turn off boundary noise (dense only)
    GPU = 1  # if 1, runs on GPU (requires CUDA), if 0, runs on CPU (slow!)
    TB = 0  # if 1, save everything to tensorboard as well as to file, if 0, just save outputs to file
else:
    with open('batch_parameters.pkl', 'rb') as f:
        inputs = pickle.load(f)
    # architecture
    model = inputs['model'][run]
    filters = inputs['filters'][run]
    filter_size = inputs['filter_size'][run]
    layers = inputs['layers'][run]
    bound_type = inputs['bound_type'][run]
    boundary_layers = inputs['boundary_layers'][run]
    softmax_temperature = inputs['softmax_temperature'][run]

    # training
    training_data = inputs['training_data'][run]
    training_batch = int(inputs['training_batch'][run])
    sample_batch_size = inputs['sample_batch_size'][run]
    n_samples = inputs['n_samples'][run]
    run_epochs = inputs['run_epochs'][run]
    dataset_size = inputs['dataset_size'][run]
    train_margin = inputs['train_margin'][run]
    average_over = int(inputs['average_over'][run])
    outpaint_ratio = inputs['outpaint_ratio'][run]
    noise = inputs['noise'][run]
    den_var = inputs['den_var'][run]
    GPU = inputs['GPU'][run]
    TB = inputs['TB'][run]

if GPU == 1:
    backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

dir_name = get_dir_name(model, training_data, filters, layers, filter_size, noise, den_var, dataset_size)  # get directory name for I/O
writer = SummaryWriter('logfiles/'+dir_name[:])  # initialize tensorboard writer

prev_epoch = 0
if __name__ == '__main__':  # run it!
    net, conv_field, optimizer, sample_0, input_x_dim, input_y_dim, sample_x_dim, sample_y_dim = initialize_training(model, filters, filter_size, layers, den_var, training_data, outpaint_ratio, dataset_size)
    net, optimizer, prev_epoch = load_checkpoint(net, optimizer, dir_name, GPU, prev_epoch)
    channels = sample_0.shape[1]
    out_maps = len(np.unique(sample_0)) + 1

    input_analysis = analyse_inputs(training_data, out_maps, dataset_size) # analyse inputs to prepare accuracy metrics

    if prev_epoch == 0: # if we are just beginning training, save inputs and relevant analysis
        save_outputs(dir_name, n_samples, layers, filters, bound_type, input_x_dim, noise, den_var, filter_size, sample_0, 1, TB)

    print('Imported and Analyzed Training Dataset {}'.format(training_data))

    if GPU == 1:
        net = nn.DataParallel(net) # go to multi-GPU training
        print("Using", torch.cuda.device_count(), "GPUs")
        net.to(torch.device("cuda:0"))
        print(summary(net, (channels, input_x_dim, input_y_dim)))  # doesn't work on CPU, not sure why

    max_epochs = run_epochs + prev_epoch + 1

    ## BEGIN TRAINING/GENERATION
    if run_epochs == 0:  # no training, just samples
        prev_epoch += 1
        epoch = prev_epoch

        # to a test of the net to get it warmed up
        training_batch, changed = get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, out_maps, channels, den_var, dataset_size, GPU)  # confirm we can keep on at this batch size
        if changed == 1:  # if the training batch is different, we have to adjust our batch sizes and dataloaders
            tr, te = get_dataloaders(training_data, training_batch, dataset_size)
            print('Training batch set to {}'.format(training_batch))
        else:
            tr, te = get_dataloaders(training_data, training_batch, dataset_size)

        sample, time_ge, n_samples, agreements, output_analysis = generation(dir_name, input_x_dim, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, noise, den_var, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature, dataset_size, GPU, cuda, TB)

    else: #train it!
        epoch = prev_epoch + 1
        converged = 0
        tr_err_hist = []
        te_err_hist = []
        while (epoch <= (max_epochs + 1)) & (converged == 0):#for epoch in range(prev_epoch+1, max_epochs):  # over a certain number of epochs

            training_batch, changed = get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, out_maps, channels, den_var, dataset_size, GPU)  # confirm we can keep on at this batch size
            if changed == 1: # if the training batch is different, we have to adjust our batch sizes and dataloaders
                tr, te = get_dataloaders(training_data, training_batch, dataset_size)
                print('Training batch set to {}'.format(training_batch))
            else:
                tr, te = get_dataloaders(training_data, training_batch, dataset_size)

            err_tr, time_tr = train_net(net, optimizer, writer, tr, epoch, out_maps, noise, den_var, conv_field, GPU, cuda)  # train & compute loss
            err_te, time_te = test_net(net, writer, te, out_maps, noise, den_var, epoch, conv_field, GPU, cuda)  # compute loss on test set
            tr_err_hist.append(torch.mean(torch.stack(err_tr)))
            te_err_hist.append(torch.mean(torch.stack(err_te)))
            print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_tr, time_te))

            save_ckpt(epoch, net, optimizer, dir_name[:]) #save checkpoint

            converged = auto_convergence(train_margin, average_over, epoch, prev_epoch, net, optimizer, dir_name, tr_err_hist, te_err_hist, max_epochs)

            epoch += 1

        tr, te = get_dataloaders(training_data, 4, 100)  # get something from the dataset
        example = next(iter(tr)).cuda()  # get seeds from test set
        raw_out = net(example[0:2, :, :, :].float())
        raw_out = raw_out[0].unsqueeze(1)
        raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
        raw_grid = raw_grid[0].cpu().detach().numpy()
        np.save('raw_outputs/' + dir_name[:], raw_grid)

        # generate samples
        sample, time_ge, n_samples, agreements, output_analysis = generation(dir_name, input_x_dim, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, noise, den_var, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature, dataset_size, GPU, cuda, TB)



'''
# show final predictions from training for a specific example
tr, te = get_dataloaders(training_data, 4, 100)  # 1 - do the accuracy analysis
example = next(iter(te)).cuda()  # get seeds from test set
raw_out = net(example[0:2,:,:,:].float())
raw_out = raw_out[0].unsqueeze(1)
raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
raw_grid = raw_grid[0].cpu().detach().numpy()
plt.figure()
plt.imshow(raw_grid)

plt.figure()
plt.imshow(example[0,0,:,:].cpu())

# tensorboard command for windows - just make sure the directory is correct
# tensorboard --logdir=logfiles/ --host localhost

radial_fourier = outputs0['radial fourier']
radial_correlation = outputs0['radial correlation']
radial_fourier_out = outputs['radial fourier']
radial_correlation_out = outputs['radial correlation']

correlation = np.zeros(len(radial_correlation))
fourier = np.zeros(len(radial_fourier))
correlation_out = np.zeros(len(radial_correlation))
fourier_out = np.zeros(len(radial_fourier))
run = int(len(radial_correlation)/10)
for i in range(len(radial_fourier)):
    if i < run:
        fourier[i] = np.average(radial_fourier[0:i])
        fourier_out[i] = np.average(radial_fourier_out[0:i])
    else:
        fourier[i] = np.average(radial_fourier[i-run:i])
        fourier_out[i] = np.average(radial_fourier_out[i-run:i])

for i in range(len(radial_correlation)):
    if i < run:
        correlation[i] = np.average(radial_correlation[0:i])
        fourier_out[i] = np.average(radial_correlation_out[0:i])
    else:
        correlation[i] = np.average(radial_correlation[i-run:i])
        correlation_out[i] = np.average(radial_correlation_out[i-run:i])

plt.subplot(2,3,1)
plt.imshow(outputs0['sample transform'])
plt.subplot(2,3,2)
plt.imshow(outputs0['sample transform'])
plt.subplot(2,3,3)
plt.plot(fourier)
plt.plot(fourier_out)
plt.subplot(2,3,4)
plt.imshow(outputs0['density correlation'])
plt.subplot(2,3,5)
plt.imshow(outputs['density correlation'])
plt.subplot(2,3,6)
plt.plot(correlation)
plt.plot(correlation_out)

n_samples = 1
maxrange = 5
bins = np.arange(-maxrange, maxrange,2 * maxrange / 25)
grid_in = np.expand_dims(sample[0,:,:,:].cpu().detach().numpy(),0)
n_particles = np.sum(grid_in != 0)
delta = bins[1]-bins[0]
re_coords = []
for n in range(n_samples):
    re_coords.append([])

for n in range(n_samples):
    for i in range(grid_in.shape[-2]):
        for j in range(grid_in.shape[-1]):
            if grid_in[n,0,i,j] != 0:
                re_coords[n].append((i - grid_in[n,0,i,j] * delta + maxrange, j - grid_in[n,1,i,j] * delta + maxrange))

new_coords2 = np.zeros((n_samples,n_particles,2))
for n in range(n_samples):
    for m in range(len(re_coords[n])):
        new_coords2[n,m,:] = re_coords[n][m] # the reconstructed coordinates
        
        

'''