from torch import nn, optim, cuda, backends
from utils import *
from models import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

run = get_input()
## parameters
if run == -1:  # user-defined parameters
    # architecture
    model = 4 # 1 is PixelCNN without residuals, 2 is PixelCNN with pre-activated residual bottlenecks, 3 is PixelDRN, 4 - DensePixelCNN
    filters = 6 # number of convolutional filters or feature maps (at initial layer for model ==3)
    out_maps = 3 # number of output logits - must equal number of possible outputs, i.e. 1 and 0
    filter_size = 7  # initial layer convolution size
    dilation = 1  # dilation scale - see model
    layers = 10 # number of hidden convolutional layers in models 1 & 2, number of layers per block (6 blocks + stem) in model 3
    bound_type = 0  # type of boundary layer, 0 = empty, 1 = seed in top left only, 2 = seed + random noise with appropriate density, 3 = seed + generated, 4 = multiple seeds
    boundary_layers = 0 # number of layers of conv_fields between sample and boundary
    grow_filters = 0  # 0 - f_maps constant throughout network, 1 - f_maps will increase deeper in the network only for model 3

    # training
    training_data = 8 # select training set: 1 - repulsive, 2 - annealed, 3 - 64x64 finite T, 4 - single brane, 5- refined branes, 6 - synthetic drying, age =1, 7 - synthetic drying, age = -1, 8- small sample synthetic drying age = -1, 9-MAC test image, 10 - graphene
    training_batch = 1024 * (2 ** 1) # size of training and test batches - it will try to run at this size, bu5t if it doesn't fit it will go smaller
    sample_batch_size = 16 # batch size for sample generator - will auto-adjust to be sm/eq than training_batch
    n_samples = sample_batch_size  # total samples to be generated when we generate, must not be zero (it may make more if there is available memory)
    run_epochs = 1000 # number of incremental epochs which will be trained over - if zero, will run just the generator
    outpaint_ratio = 1 # sqrt of size of output relative to input
    noise = 0 # mean of border noise
    den_var = 1 # standard deviation of density variation, also toggle 0 to turn off boundary noise (dense only)
    GPU = 1  # if 1, runs on GPU (requires CUDA), if 0, runs on CPU (slow!)
    TB = 0  # if 1, save everything to tensorboard as well as to file, if 0, just save outputs to file
else:
    with open('batch_parameters.pkl', 'rb') as f:
        inputs = pickle.load(f)
    # architecture
    model = inputs['model'][run]
    filters = inputs['filters'][run]
    out_maps = inputs['out_maps'][run]
    filter_size = inputs['filter_size'][run]
    dilation = inputs['dilation'][run]
    layers = inputs['layers'][run]
    bound_type = inputs['bound_type'][run]
    boundary_layers = inputs['boundary_layers'][run]
    grow_filters = inputs['grow_filters'][run]

    # training
    training_data = inputs['training_data'][run]
    training_batch = int(inputs['training_batch'][run])
    sample_batch_size = inputs['sample_batch_size'][run]
    n_samples = inputs['n_samples'][run]
    run_epochs = inputs['run_epochs'][run]
    outpaint_ratio = inputs['outpaint_ratio'][run]
    noise = inputs['noise'][run]
    den_var = inputs['den_var'][run]
    GPU = inputs['GPU'][run]
    TB = inputs['TB'][run]

if GPU == 1:
    backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

dir_name = get_dir_name(model, training_data, filters, layers, dilation, grow_filters, filter_size, noise, den_var)  # get directory name for I/O
writer = SummaryWriter('logfiles/'+dir_name[:])  # initialize tensorboard writer

prev_epoch = 0
if __name__ == '__main__':  # run it!
    net, conv_field, optimizer, sample_0, input_x_dim, input_y_dim, sample_x_dim, sample_y_dim = initialize_training(model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var, training_data, outpaint_ratio)
    net, optimizer, prev_epoch = load_checkpoint(net, optimizer, dir_name, GPU, prev_epoch)

    #density, en_dist, correlation2d, radial_correlation, fourier2d, radial_fourier, sum, variance = analyse_inputs(training_data, out_maps, GPU)
    input_analysis = analyse_inputs(training_data, out_maps, GPU)

    if prev_epoch == 0: # save inputs and relevant analysis
        save_outputs(dir_name, n_samples, layers, filters, dilation, bound_type, input_x_dim, noise, den_var, filter_size, sample_0, 1, TB)

    print('Imported and Analyzed Training Dataset {}'.format(training_data))

    if GPU == 1:
        net = nn.DataParallel(net)
        print("Using", torch.cuda.device_count(), "GPUs")
        net.to(torch.device("cuda:0"))
        print(summary(net, (1, input_x_dim, input_y_dim)))  # doesn't work on CPU, not sure why

    max_epochs = run_epochs + prev_epoch + 1

    if run_epochs == 0:  # no training, just samples
        prev_epoch += 1
        epoch = prev_epoch

        generator = get_generator(model, filters, filter_size, dilation, layers, out_maps, grow_filters, 0, GPU, net)
        sample, time_ge, sample_batch_size, n_samples = generate_samples(n_samples, sample_batch_size, sample_x_dim, sample_y_dim, conv_field, generator, bound_type, GPU, cuda, training_data, out_maps, boundary_layers, noise, den_var)  # generate samples

        del generator
        if n_samples != 0:
            save_ckpt(prev_epoch, net, optimizer, dir_name)
            print('Generated samples')

            output_analysis = analyse_samples(sample, training_data)
            save_outputs(dir_name, n_samples, layers, filters, dilation, bound_type, input_x_dim, noise, den_var, filter_size, sample, epoch, TB)

            agreements = compute_accuracy(input_analysis, output_analysis, outpaint_ratio, training_data)
            total_agreement = 1
            for i, j, in enumerate(agreements.values()):
                total_agreement *= float(j)

            if training_data == 9:
                print('tot = {:.4f}; den={:.2f}; b_order={:.2f}; b_length={:.2f}; b_angle={:.2f}; corr={:.2f}; fourier={:.2f}; time_ge={:.1f}s'.format(total_agreement, agreements['density'], agreements['order'], agreements['bond'], agreements['angle'], agreements['correlation'], agreements['fourier'], time_ge))
            else:
                print('tot = {:.4f}; den={:.2f}; en={:.2f}; corr={:.2f}; fourier={:.2f}; time_ge={:.1f}s'.format(total_agreement, agreements['density'], agreements['energy'], agreements['correlation'], agreements['fourier'], time_ge))

    else: #train it!
        epoch = prev_epoch + 1
        converged = 0
        tr_err_hist = []
        te_err_hist = []
        while (epoch <= (max_epochs + 1)) & (converged == 0):#for epoch in range(prev_epoch+1, max_epochs):  # over a certain number of epochs

            training_batch, changed = get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var, GPU)  # confirm we can keep on at this batch size
            if changed == 1: # if the training batch is different, we have to adjust our batch sizes and dataloaders
                tr, te = get_dataloaders(training_data, training_batch, out_maps)
                print('Training batch set to {}'.format(training_batch))
            else:
                tr, te = get_dataloaders(training_data, training_batch, out_maps)

            err_tr, time_tr = train_net(net, optimizer, writer, tr, epoch, out_maps, noise, den_var, conv_field, GPU, cuda)  # train & compute loss
            err_te, time_te = test_net(net, writer, te, out_maps, noise, den_var, epoch, conv_field, GPU, cuda)  # compute loss on test set
            tr_err_hist.append(torch.mean(torch.stack(err_tr)))
            te_err_hist.append(torch.mean(torch.stack(err_te)))
            print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_tr, time_te))

            average_over = 5
            converged = auto_convergence(average_over, epoch, prev_epoch, net, optimizer, dir_name, tr_err_hist, te_err_hist, max_epochs)

            if converged == 1:  # if we are sampling
                net, conv_field = get_model(model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var)
                optimizer = optim.Adam(net.parameters())#optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#
                net, optimizer, prev_epoch = load_checkpoint(net, optimizer, dir_name, GPU, prev_epoch)
                if GPU == 1:
                    net = nn.DataParallel(net)
                    net.to(torch.device("cuda:0"))

                del tr, te
                generator = get_generator(model, filters, filter_size, dilation, layers, out_maps, grow_filters, 0, GPU, net)
                sample, time_ge, sample_batch_size, n_samples = generate_samples(n_samples, sample_batch_size, sample_x_dim, sample_y_dim, conv_field, generator, bound_type, GPU, cuda, training_data, out_maps, boundary_layers, noise, den_var)  # generate samples
                if n_samples != 0:
                    save_ckpt(prev_epoch, net, optimizer, dir_name)
                    print('Generated samples')

                    output_analysis = analyse_samples(sample, training_data)
                    save_outputs(dir_name, n_samples, layers, filters, dilation, bound_type, input_x_dim, noise, den_var, filter_size, sample, epoch, TB)

                    agreements = compute_accuracy(input_analysis, output_analysis, outpaint_ratio, training_data)
                    total_agreement = 1
                    for i, j, in enumerate(agreements.values()):
                        total_agreement *= float(j)

                    if training_data == 9:
                        print('tot = {:.4f}; den={:.2f}; b_order={:.2f}; b_length={:.2f}; b_angle={:.2f}; corr={:.2f}; fourier={:.2f}; time_ge={:.1f}s'.format(total_agreement, agreements['density'], agreements['order'], agreements['bond'], agreements['angle'], agreements['correlation'], agreements['fourier'], time_ge))
                    else:
                        print('tot = {:.4f}; den={:.2f}; en={:.2f}; corr={:.2f}; fourier={:.2f}; time_ge={:.1f}s'.format(total_agreement, agreements['density'], agreements['energy'], agreements['correlation'], agreements['fourier'], time_ge))


            if converged == 0:
                if epoch == max_epochs -1:
                    save_ckpt(epoch, net, optimizer, dir_name[:])

            epoch += 1

        tr, te = get_dataloaders(training_data, 4, out_maps)  # 1 - do the accuracy analysis
        example = next(iter(te)).cuda()  # get seeds from test set
        raw_out = net(example[0:2, :, :, :].float())
        raw_out = raw_out[0].unsqueeze(1)
        raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
        raw_grid = raw_grid[0].cpu().detach().numpy()
        np.save('raw_outputs/' + dir_name[:], raw_grid)


'''
# show final predictions from training for a specific example
tr, te = get_dataloaders(training_data, 4, out_maps)  # 1 - do the accuracy analysis
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

'''