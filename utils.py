# metrics to determine the performance of our learning algorithm
import numpy as np
import torch.nn.functional as F
import os
from torch import nn, optim, cuda, backends
import torch
from torch.utils import data
import time
import pickle
from torch.utils.data import Dataset
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
import sys
import tqdm
from accuracy_metrics import *
from models import *
from Image_Processing_Utils import *
import argparse


def get_input():
    parser = argparse.ArgumentParser()  # parse run index so we can parallelize submission
    parser.add_argument('--run_num', type=int, default = -1)
    cmd_line_input = parser.parse_args()
    run = cmd_line_input.run_num

    return run

class build_dataset(Dataset):
    def __init__(self, training_data, out_maps):
        if training_data == 1:
            self.samples = np.load('data/repulsive_redo_configs2.npy', allow_pickle=True).astype('uint8')
            self.samples = np.expand_dims(self.samples, axis=1)
        elif training_data == 2:
            self.samples = np.load('data/annealment_redo_configs.npy', allow_pickle=True).astype('uint8')
            self.samples = np.transpose(self.samples, [2,1,0])
            self.samples = np.expand_dims(self.samples, axis=1)
        elif training_data == 3:
            #self.samples = np.load('data/sparse_64x64_configs.npy',allow_pickle=True).astype('uint8')
            #self.samples = np.expand_dims(self.samples, axis=1)
            self.samples = np.load('Finite_T_Sample.npy',allow_pickle=True).astype('uint8')
        elif training_data == 4:
            self.samples = np.load('Augmented_Brain_Sample.npy',allow_pickle=True).astype('uint8')
        elif training_data == 5:
            self.samples = np.load('Augmented_Brain_Sample2.npy',allow_pickle=True).astype('uint8')
        elif training_data == 6:
            self.samples = np.load('drying_sample_1.npy', allow_pickle=True)
        elif training_data == 7:
            self.samples = np.load('drying_sample_-1.npy', allow_pickle=True)
        elif training_data == 8:
            self.samples = np.load('big_worm_results.npy', allow_pickle=True)
        elif training_data == 9:
            self.samples = np.load('data/MAC/big_MAC.npy',allow_pickle=True)
        elif training_data == 10:
            self.samples = np.load('data/MAC/graphene.npy',allow_pickle=True)

        self.samples = np.array((self.samples[0:10000] + 1)/(out_maps - 1)) # normalize inputs on 0,1,2...

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dir_name(model, training_data, filters, layers, dilation, grow_filters, filter_size, noise, den_var):
    if model == 3:
        dir_name = "model=%d_dataset=%d_filters=%d_layers=%d_dilation=%d_grow_filters=%d_filter_size=%d_noise=%.1f_denvar=%.1f" % (model, training_data, filters, layers, dilation, grow_filters, filter_size, noise, den_var)  # directory where tensorboard logfiles will be saved
    else:
        dir_name = "model=%d_dataset=%d_filters=%d_layers=%d_filter_size=%d_noise=%.1f_denvar=%.1f" % (model, training_data, filters, layers, filter_size, noise, den_var)  # directory where tensorboard logfiles will be saved

    return dir_name

def get_model(model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var):
    if model == 1:
        net = PixelCNN(filters, filter_size, layers, out_maps, 1)  # 1 means convolutions will be padded
        conv_field = (filter_size - 1) * layers // 2  # range of convolutional receptive field for given model - for PixelCNN
    elif model == 2:
        net = PixelCNN_RES(filters, filter_size, layers, out_maps, 1)
        conv_field = layers + (filter_size - 1) // 2  # for PixelCNN_RES
    elif model == 3:
        net = PixelDRN(filters, filter_size, dilation, layers, out_maps, grow_filters, 1)
        conv_field = int(np.sum(net.block_dilation[1:])) * layers + (filter_size - 1) // 2  # conv field is equal to all the unpadding we will have to do for residuals in generation
    elif model == 4:
        net = DensePixelDCNN(filters,filter_size,dilation,layers,out_maps, den_var == 0)
        conv_field = layers + (filter_size - 1) // 2

    def init_weights(m):
        if (type(m) == nn.Conv2d) or (type(m) == MaskedConv2d):
            #torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.kaiming_uniform_(m.weight)

    net.apply(init_weights) # apply xavier weights to 1x1 and 3x3 convolutions

    return net, conv_field


def get_dataloaders(training_data, training_batch, out_maps):
    dataset = build_dataset(training_data, out_maps)  # get data
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # randomly split the data into training and test sets
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    tr = data.DataLoader(train_dataset, batch_size=training_batch, shuffle=True, num_workers= 0, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers= 0, pin_memory=True)

    return tr, te


def initialize_training(model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var, training_data, outpaint_ratio):
    net, conv_field = get_model(model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var)
    optimizer = optim.Adam(net.parameters()) #optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#
    tr, te = get_dataloaders(training_data, 4, out_maps)
    sample_0 = next(iter(tr))
    input_x_dim, input_y_dim = sample_0.shape[-1], sample_0.shape[-2]  # set input and output dimensions
    sample_x_dim, sample_y_dim = int(input_x_dim * outpaint_ratio), int(input_y_dim * outpaint_ratio)

    return net, conv_field, optimizer, sample_0, input_x_dim, input_y_dim, sample_x_dim, sample_y_dim


def load_checkpoint(net, optimizer, dir_name, GPU, prev_epoch):
    if os.path.exists('ckpts/' + dir_name[:]):  #reload model
        checkpoint = torch.load('ckpts/' + dir_name[:])

        if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
            for i in list(checkpoint['model_state_dict']):
                checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']

        if GPU == 1:
            net.cuda()  # move net to GPU
            for state in optimizer.state.values():  # move optimizer to GPU
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        net.eval()
        print('Reloaded model: ', dir_name[:])
    else:
        print('New model: ', dir_name[:])

    return net, optimizer, prev_epoch


def get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var, GPU):
    net, conv_field = get_model(model, filters, filter_size, layers, out_maps, grow_filters, dilation, den_var)
    if GPU == 1:
        net = nn.DataParallel(net)
        net.to(torch.device("cuda:0"))

    optimizer =  optim.Adam(net.parameters())
    finished = 0
    training_batch_0 = 1 * training_batch
    #  test various batch sizes to see what we can store in memory
    test_dataset = build_dataset(training_data, out_maps)
    while (training_batch > 1) & (finished == 0):
        try:
            net.train(True)
            test_dataloader = data.DataLoader(test_dataset, batch_size = training_batch, shuffle=False, num_workers = 0, pin_memory = True)
            input = next(iter(test_dataloader))
            if GPU == 1:
                input = input.cuda()

            target = input.data[:, 0] * (out_maps - 1)  # switch from training to output space

            noise = 0
            if den_var != 0:
                input = random_padding(input, noise, den_var, conv_field, GPU)

            loss = F.cross_entropy(net(input.float()), target.long())
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters
            finished = 1
        except RuntimeError: # if we get an OOM, try again with smaller batch
            training_batch = training_batch // 2

    return int(np.ceil(training_batch * .8)), int(training_batch != training_batch_0)


def train_net(net, optimizer, writer, tr, epoch, out_maps, noise, den_var, conv_field, GPU, cuda):
    if GPU == 1:
        cuda.synchronize()  # synchronize for timing purposes
    time_tr = time.time()

    err_tr = []
    net.train(True)
    for i, input in enumerate(tr):
        if GPU == 1:
            input = input.cuda(non_blocking=True)

        target = input.data[:, 0] * (out_maps - 1)  # switch from training to output space

        #if noise != 0:
            #input = scramble_images(input, noise, den_var, GPU)

        if den_var !=0:
            input = random_padding(input, noise, den_var, conv_field, GPU)

        loss = F.cross_entropy(net(input.float()), target.long())  # compute the loss between the network output, and our target
        err_tr.append(loss.data)  # record loss
        optimizer.zero_grad()  # reset gradients from previous passes
        loss.backward()  # back-propagation
        optimizer.step()  # update parameters

        if i % 10 == 0:  # log loss to tensorboard
            writer.add_scalar('training_loss', loss.data, epoch * len(tr) + i)

    if GPU == 1:
        cuda.synchronize()
    time_tr = time.time() - time_tr

    return err_tr, time_tr


def test_net(net, writer, te, out_maps, noise, den_var, epoch, conv_field, GPU, cuda):
    if GPU == 1:
        cuda.synchronize()

    time_te = time.time()
    err_te = []
    net.train(False)
    with torch.no_grad():  # we're just computing the test set error so we won't be updating the gradients or weights
        for i, input in enumerate(te):
            if GPU == 1:
                input = input.cuda()

            target = input.data[:, 0] * (out_maps - 1)  # switch from training to output space

            #if noise != 0:
            #    input = scramble_images(input, noise, den_var, GPU)

            if den_var != 0:
                input = random_padding(input, noise, den_var, conv_field, GPU)

            loss = F.cross_entropy(net(input.float()), target.long())
            err_te.append(loss.data)

            if i % 10 == 0:  # log loss to tensorboard
                writer.add_scalar('test_loss', loss.data, epoch * len(te))  # writer.add_histogram('conv1_weight', net[0].weight[0], epoch)  # if you want to watch the evolution of the filters  # writer.add_histogram('conv1_grad', net[0].weight.grad[0], epoch)

    if GPU == 1:
        cuda.synchronize()
    time_te = time.time() - time_te

    return err_te, time_te

def auto_convergence(average_over, epoch, prev_epoch, net, optimizer, dir_name, tr_err_hist, te_err_hist, max_epochs):
    # set convergence criteria
    # if the test error has increased on average for the last x epochs
    # or if the training error has decreased by less than 1% for the last x epochs
    train_margin = 0.0001  # relative change over past x runs
    # or if the training error is diverging from the test error by more than 20%
    test_margin = 10# 0.1
    # average_over - the time over which we will average loss in order to determine convergence
    converged = 0
    if (epoch - prev_epoch) <= average_over:  # early checkpointing
        save_ckpt(epoch, net, optimizer, dir_name[:] +'_ckpt_-{}'.format(average_over - (epoch - prev_epoch)))

    if (epoch - prev_epoch) > average_over:
        os.remove('ckpts/'+dir_name[:]+'_ckpt_-{}'.format(average_over - 1))  # delete trailing checkpoint
        for i in range(average_over - 2, -1, -1):  # move all previous checkpoints
            os.rename('ckpts/'+dir_name[:]+'_ckpt_-{}'.format(i), 'ckpts/'+dir_name[:]+'_ckpt_-{}'.format(i + 1))
        save_ckpt(epoch, net, optimizer, dir_name[:]+'_ckpt_-{}'.format(0))  # save new checkpoint

        tr_mean, te_mean = [torch.mean(torch.stack(tr_err_hist[-average_over:])), torch.mean(torch.stack(te_err_hist[-average_over:]))]
        if (te_mean > te_err_hist[-average_over]) or (torch.abs((tr_mean - tr_err_hist[-average_over]) / tr_mean) < train_margin) or (((te_mean - tr_mean) / tr_mean) > test_margin) or ((epoch - prev_epoch) == max_epochs):
            converged = 1
            if os.path.exists('ckpts/'+dir_name[:]) & (epoch > 1) & (epoch-prev_epoch < average_over): #can't happen on first epoch
                print('Previously converged this result at epoch {}!'.format(epoch - average_over -1))
            else:
                if os.path.exists('ckpts/'+dir_name[:]):
                    os.remove('ckpts/'+dir_name[:])
                os.rename('ckpts/'+dir_name[:]+'_ckpt_-{}'.format(average_over - 1), 'ckpts/'+dir_name[:])  # save the -average_over epoch as the final output
                print('Learning converged at epoch {}'.format(epoch - average_over + 1))  # print a nice message  # consider also using an accuracy metric

    return converged


def get_generator(model, filters, filter_size, dilation, layers, out_maps, grow_filters, padding, GPU, net):
    if model == 1:
        generator = PixelCNN(filters, filter_size, layers, out_maps, padding)  # 0 means no padding
    elif model == 2:
        generator = PixelCNN_RES_OUT(filters, filter_size, layers, out_maps, padding)
    elif model == 3:
        generator = PixelDRN(filters, filter_size, dilation, layers, out_maps, grow_filters, padding)
    elif model == 4:
        generator = DensePixelDCNN(filters, filter_size, dilation, layers, out_maps, padding)

    if GPU == 1:
        generator = nn.DataParallel(generator)
        generator.to(torch.device("cuda:0"))

    generator.load_state_dict(net.state_dict())

    return generator


def build_boundary(sample_batch, sample_batch_size, training_data, conv_field, generator, bound_type, out_maps, noise_mean, den_var, GPU): # 0 = empty, 1 = seed in top left only, 2 = seed + random noise with appropriate density, 3 = seed + generated

    if bound_type > 0:  # requires samples are at least as large as the convolutional receptive field, and
        tr, te = get_dataloaders(training_data, int(sample_batch_size/.2), out_maps) # requires a sufficiently large training set or we won't saturate the seeds
        seeds = next(iter(tr))  # get seeds from training set

        if (bound_type == 1) or (bound_type == 3):
            sample_batch[:, :, 0:conv_field, 0:seeds.shape[3]] = seeds[0:sample_batch_size, :, 0:conv_field, 0:np.amin((seeds.shape[3], sample_batch.shape[3]))]  # seed from the top-left
            sample_batch[:, :, 0:seeds.shape[2], 0:conv_field] = seeds[0:sample_batch_size, :, 0:np.amin((seeds.shape[2], sample_batch.shape[2])), 0:conv_field]  # seed from the top-left

        elif bound_type == 4: # a bunch of seeds around the top and sides of the sample #DEPRECATED
            sample_batch[:, :, 0:conv_field, 0:seeds.shape[3]] = seeds[0:sample_batch_size, :, 0:conv_field, 0:seeds.shape[3]]  # seed from the top-left
            sample_batch[:, :, 0:seeds.shape[2], 0:conv_field] = seeds[0:sample_batch_size, :, 0:seeds.shape[2], 0:conv_field]  #
            for i in range(sample_batch.shape[3]//seeds.shape[3] - 1): # seed the top row
                sample_batch[:, :, 0:conv_field, i*seeds.shape[3]:(i+1)*seeds.shape[3]] = seeds[torch.randint(0, seeds.shape[0], (sample_batch_size,)), :, 0:conv_field, 0:seeds.shape[3]]

            for i in range(sample_batch.shape[2]//seeds.shape[2] - 1): # seed the sides
                sample_batch[:, :, i*seeds.shape[2]:(i+1)*seeds.shape[2], 0:conv_field] = seeds[torch.randint(0, seeds.shape[0], (sample_batch_size,)), :, 0:seeds.shape[2], 0:conv_field]
                sample_batch[:, :, i*seeds.shape[2]:(i+1)*seeds.shape[2], -conv_field:] = seeds[torch.randint(0, seeds.shape[0], (sample_batch_size,)), :, 0:seeds.shape[2], 0:conv_field]


    if bound_type == 2: # fill total image with random noise at appropriate density ##fill unseeded boundary area with random noise of appropriate density - binary output only
        #noise1 = torch.Tensor(np.random.choice((0,1), size = (sample_batch_size, 1, conv_field, sample_batch.shape[3]-seeds.shape[3]) , p = (1 - density, density))) # fill for upper bound
        #noise2 = torch.Tensor(np.random.choice((0,1), size = (sample_batch_size, 1, sample_batch.shape[2] - seeds.shape[2], conv_field), p = (1 - density, density))) # fill for left-bound

        #sample_batch[:, :, 0:conv_field, seeds.shape[3]:] = noise1
        #sample_batch[:, :, seeds.shape[2]:, 0:conv_field] = noise2
        #density = torch.mean(seeds.type(torch.DoubleTensor))
        #sample_batch = torch.Tensor(np.random.choice((0,1), size = (sample_batch_size, 1, sample_batch.shape[2], sample_batch.shape[3]), p = (1 - density, density)))
        sample_batch = torch.Tensor(np.random.normal(noise_mean, den_var, size =(sample_batch_size, 1, sample_batch.shape[2], sample_batch.shape[3])))

    elif bound_type == 3: # fill unseeded boundary using the network #DEPRECATED
        bound1 = sample_batch[:, :, 0:conv_field, 0:] # take horizontal and vertical blocks
        bound1 = bound1.transpose(2,3) # flip so both are vertically oriented
        bound2 = sample_batch[:, :, 0:, 0:conv_field]
        # pad the sides appropriately with the conv_field
        bound1 = F.pad(bound1, (conv_field, conv_field, 0, conv_field), mode = 'constant', value = 0) # manual padding is necessary for how we've set up the generator
        bound2 = F.pad(bound2, (conv_field, conv_field, 0, conv_field), mode = 'constant', value = 0)

        if GPU == 1:
            bound1 = bound1.cuda()
            bound2 = bound2.cuda()

        sample1_x_dim, sample1_y_dim, sample2_x_dim, sample2_y_dim = [bound1.shape[3], bound1.shape[2]-conv_field, bound2.shape[3], bound2.shape[2]-conv_field]

        generator.train(False)
        with torch.no_grad():  # we will not be updating weights
            for i in range(seeds.shape[3], sample1_y_dim):  # for each pixel
                for j in range(conv_field, conv_field * 2 ):
                    out = generator(bound1[:, :, i - conv_field:i + conv_field + 1, j - conv_field:j + conv_field + 1])  # query the network about only area within the receptive field
                    probs = F.softmax(out[:, :, 0, 0], dim=1).data
                    bound1[:, :, i, j] = torch.multinomial(probs, 1).float()  # / (out_maps-1)  # sample from the above probability distribution

            for i in range(seeds.shape[2], sample2_y_dim):  # for each pixel
                for j in range(conv_field, conv_field * 2 ):
                    out = generator(bound2[:, :, i - conv_field:i + conv_field + 1, j - conv_field:j + conv_field + 1])  # query the network about only area within the receptive field
                    probs = F.softmax(out[:, 0:3:2, 0, 0], dim=1).data
                    bound2[:, :, i, j] = (torch.multinomial(probs, 1).float() - 0.5) *(out_maps-1)  # sample from the above probability distribution

        bound1 = F.pad(bound1, (-conv_field, -conv_field, 0, -conv_field), mode='constant', value=0)  # unpad both bounds
        bound2 = F.pad(bound2, (-conv_field, -conv_field, 0, -conv_field), mode='constant', value=0)
        bound1 = bound1.transpose(2,3) # flip bound1 back to horizontal

        sample_batch[:, :, 0:conv_field, seeds.shape[3]:] = bound1[:, :, 0:conv_field, seeds.shape[3]:]  # assign bounds
        sample_batch[:, :, seeds.shape[2]:, 0:conv_field] = bound2[:, :, seeds.shape[2]:, 0:conv_field]


    return sample_batch

def get_sample_batch_size(sample_batch_size, generator, sample_x_dim, sample_y_dim, conv_field, GPU):
    # dynamically set sample batch size
    finished = 0
    sample_batch_size_0 = 1 * sample_batch_size
    #  test various batch sizes to see what we can store in memory
    while (sample_batch_size > 1) & (finished == 0):
        try:
            input = torch.Tensor(sample_batch_size, 1, sample_y_dim + 2 * conv_field, sample_x_dim + 2 * conv_field)
            if GPU == 1:
                input = input.cuda()

            test_out = generator(input[:, :, conv_field - conv_field:conv_field + conv_field + 1, conv_field - conv_field:conv_field + conv_field + 1].float())
            finished = 1
        except RuntimeError:  # if we get an OOM, try again with smaller batch
            sample_batch_size = sample_batch_size // 2

    return sample_batch_size, int(sample_batch_size != sample_batch_size_0)

def generate_samples(n_samples, sample_batch_size, sample_x_dim, sample_y_dim, conv_field, generator, bound_type, GPU, cuda, training_data, out_maps, boundary_layers, noise_mean, den_var):
    if GPU == 1:
        cuda.synchronize()
    time_ge = time.time()

    sample_x_padded = sample_x_dim + 2 * conv_field * boundary_layers
    sample_y_padded = sample_y_dim + conv_field * boundary_layers  # don't need to pad the bottom

    sample_batch_size, changed = get_sample_batch_size(sample_batch_size, generator, sample_x_padded, sample_y_padded, conv_field, GPU) # add extra padding by conv_field in both x-directions, and in the + y direction, which we will remove later
    if changed:
        print('Sample batch size changed to {}'.format(sample_batch_size))

    if n_samples < sample_batch_size:
        n_samples = sample_batch_size

    batches = int(np.ceil(n_samples/sample_batch_size))
    n_samples = sample_batch_size * batches
    sample = torch.ByteTensor(n_samples, 1, sample_y_dim, sample_x_dim)  # sample placeholder
    print('Generating {} Samples'.format(n_samples))

    for batch in range(batches):  # can't do these all at once so we do it in batches
        print('Batch {} of {} batches'.format(batch + 1, batches))
        sample_batch = torch.FloatTensor(sample_batch_size, 1, sample_y_padded + 2 * conv_field, sample_x_padded + 2 * conv_field)  # needs to be explicitly padded by the convolutional field
        sample_batch.fill_(0)  # initialize with minimum value

        if bound_type > 0:
            sample_batch = build_boundary(sample_batch, sample_batch_size, training_data, conv_field, generator, bound_type, out_maps, noise_mean, den_var, GPU)

        if GPU == 1:
            sample_batch = sample_batch.cuda()

        generator.train(False)
        with torch.no_grad():  # we will not be updating weights
            for i in tqdm.tqdm(range(conv_field, sample_y_padded + conv_field)):  # for each pixel
                for j in range(conv_field, sample_x_padded + conv_field):
                    out = generator(sample_batch[:, :, i - conv_field:i + conv_field + 1, j - conv_field:j + conv_field + 1].float())  # query the network about only area within the receptive field
                    probs = F.softmax(out[:, 1:, 0, 0], dim=1).data # the remove the lowest element (boundary)
                    sample_batch[:, :, i, j] = (torch.multinomial(probs, 1).float() + 1) / (out_maps -1)  # convert output back to training space

        del out, probs

        sample[batch * sample_batch_size:(batch + 1) * sample_batch_size, :, :, :] = sample_batch[:, :, (boundary_layers + 1) * conv_field:-conv_field, (boundary_layers + 1) * conv_field:-((boundary_layers + 1) * conv_field)]  * (out_maps - 1) - 1 # convert back to input space

    if GPU == 1:
        cuda.synchronize()
    time_ge = time.time() - time_ge

    return sample, time_ge, sample_batch_size, n_samples

def analyse_inputs(training_data, out_maps, GPU):
    dataset = torch.Tensor(build_dataset(training_data, out_maps))  # get data
    dataset = dataset * (out_maps - 1) - 1
    #avg_density, en_dist, correlation2d, radial_correlation, fourier2d, radial_fourier, sum, variance,  = sample_analysis(dataset)
    input_analysis = analyse_samples(dataset, training_data)

    return input_analysis

def analyse_samples(sample, training_data):
    # considering the highest value in the dist to be that for particles
    particles = torch.max(sample)
    avg_density = torch.mean((sample==particles).type(torch.float32)) # for A
    sum = torch.sum(sample[:,0,:,:]==particles,0)
    variance = torch.var(sum/torch.mean(sum + 1e-5))
    correlation2d, radial_correlation, correlation_bins = spatial_correlation(sample==particles)
    fourier2d = fourier_analysis(torch.Tensor((sample==particles).float()))
    fourier_bins, radial_fourier = radial_fourier_analysis(fourier2d)

    if training_data == 9:
        avg_bond_order, bond_order_dist, avg_bond_length, avg_bond_angle, bond_length_dist, bond_angle_dist = bond_analysis(sample, 1.7, particles)
    else:
        avg_interactions, en_dist = compute_interactions(sample == particles)

    sample_analysis = {}
    sample_analysis['density'] = avg_density
    sample_analysis['sum'] = sum
    sample_analysis['variance'] = variance
    sample_analysis['correlation2d'] = correlation2d
    sample_analysis['radial correlation'] = radial_correlation
    sample_analysis['correlation bins'] = correlation_bins
    sample_analysis['fourier2d'] = fourier2d
    sample_analysis['radial fourier'] = radial_fourier
    sample_analysis['fourier bins'] = fourier_bins
    if training_data == 9:
        sample_analysis['average bond order'] = avg_bond_order
        sample_analysis['bond order dist'] = bond_order_dist
        sample_analysis['average bond length'] = avg_bond_length
        sample_analysis['bond length dist'] = bond_length_dist
        sample_analysis['average bond angle'] = avg_bond_angle
        sample_analysis['bond angle dist'] = bond_angle_dist
    else:
        sample_analysis['average interactions'] = avg_interactions
        sample_analysis['interactions dist'] = en_dist

    return sample_analysis

def compute_accuracy(input_analysis, output_analysis, outpaint_ratio, training_data):
    input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['fourier2d'].shape[-1], input_analysis['fourier2d'].shape[-2], output_analysis['fourier2d'].shape[-1], output_analysis['fourier2d'].shape[-2]]

    if outpaint_ratio > 1: # shrink inputs to meet outputs or vice-versa
        x_difference = sample_xdim-input_xdim
        y_difference = sample_ydim-input_ydim
        output_analysis['fourier2d'] = output_analysis['fourier2d'][y_difference//2:-y_difference//2, x_difference//2:-x_difference//2]
    elif outpaint_ratio < 1:
        x_difference = input_xdim - sample_xdim
        y_difference = input_ydim- sample_ydim
        input_analysis['fourier2d'] = input_analysis['fourier2d'][y_difference // 2:-y_difference // 2, x_difference // 2:-x_difference // 2]

    input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['correlation2d'].shape[-1], input_analysis['correlation2d'].shape[-2], output_analysis['correlation2d'].shape[-1], output_analysis['correlation2d'].shape[-2]]
    if outpaint_ratio > 1: # shrink inputs to meet outputs or vice-versa
        x_difference = sample_xdim-input_xdim
        y_difference = sample_ydim-input_ydim
        output_analysis['correlation2d'] = output_analysis['correlation2d'][y_difference//2:-y_difference//2, x_difference//2:-x_difference//2]
    elif outpaint_ratio < 1:
        x_difference = input_xdim - sample_xdim
        y_difference = input_ydim- sample_ydim
        input_analysis['correlation2d'] = input_analysis['correlation2d'][y_difference // 2:-y_difference // 2, x_difference // 2:-x_difference // 2]

    agreements = {}
    agreements['density'] = np.amax((1 - np.abs(input_analysis['density'] - output_analysis['density']) / input_analysis['density'],0))
    agreements['fourier'] = np.amax((1 - np.sum(np.abs(input_analysis['fourier2d'] - output_analysis['fourier2d'])) / (np.sum(input_analysis['fourier2d']) + 1e-8),0))
    agreements['correlation'] = np.amax((1 - np.sum(np.abs(input_analysis['correlation2d'] - output_analysis['correlation2d'])) / (np.sum(input_analysis['correlation2d']) + 1e-8),0))

    if training_data == 9:
        agreements['order'] = np.amax((1 - np.average(np.abs(input_analysis['bond order dist'][0] - output_analysis['bond order dist'][0])) / np.average(input_analysis['bond order dist'][0]), 0))
        agreements['bond'] = np.amax((1 - np.average(np.abs(input_analysis['bond length dist'][0] - output_analysis['bond length dist'][0])) / np.average(input_analysis['bond length dist'][0]), 0))
        agreements['angle'] = np.amax((1 - np.average(np.abs(input_analysis['bond angle dist'][0] - output_analysis['bond angle dist'][0])) / np.average(input_analysis['bond angle dist'][0]), 0))
    else:
        agreements['energy'] = np.amax((1 - np.average(np.abs(input_analysis['interactions dist'][0] - output_analysis['interactions dist'][0])) / np.average(input_analysis['interactions dist'][0]), 0))

    return agreements


def write_inputs(layers, filters, max_epochs, n_samples, filter_size, writer):
    writer.add_text('layers', '%d' % layers)
    writer.add_text('feature maps', '%d' % filters)
    writer.add_text('epochs', '%d' % max_epochs)
    writer.add_text('# samples', '%d' % n_samples)
    writer.add_text('filter size', '%d' % filter_size)


def save_outputs(dir_name, n_samples, layers, filters, dilation, bound_type, input_x_dim, noise, den_var, filter_size, sample, epoch, TB):
    # save to file
    output = {}
    output['n_samples'] = n_samples
    output['layers'] = layers
    output['filters'] = filters
    output['first layer filter size'] = filter_size
    output['epoch'] = epoch
    output['dilation'] = dilation
    output['bound type'] = bound_type
    output['input size'] = input_x_dim
    output['noise mean'] = noise
    output['noise variance'] = den_var
    #output['radial correlation'] = radial_correlation
    #output['density correlation'] = correlation2d
    #output['sample transform'] = fourier2d
    #output['radial fourier'] = radial_fourier
    #output['average density'] = avg_density
    #output['interactions hist'] = interactions_hist
    #output['spatial distribution'] = sum_dist
    output['sample'] = sample

    with open('outputs/'+dir_name[:]+'_epoch=%d'%epoch+'.pkl', 'wb') as f:
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    # to load
    '''
    #with open('outputs/' + dir_name[:] + '.pkl', 'rb') as f:
        outputs = pickle.load(f)
    '''

    if TB == 1:  # save to tensorboard #DEPRECATED
        '''
        #writer.add_scalar('pooled spatial variance', pooled_variance, epoch)
        #writer.add_scalar('spatial variance', sum_variance, epoch)
        writer.add_scalar('fourier overlap', fourier_overlap, epoch)
        #writer.add_scalar('average interactions', avg_interactions, epoch)
        #writer.add_scalar('average density', avg_density, epoch)
        writer.add_scalar('density overlap', density_agreement, epoch)
        writer.add_scalar('interactions overlap', interactions_overlap, epoch)
        writer.add_scalar('variance overlap', variance_overlap, epoch)
        writer.add_image('fourier transform', np.log(image_transform)/np.log(image_transform.max()), epoch, dataformats='HW')
        writer.add_image('pooled distribution', pooled_dist, epoch, dataformats='HW')
        writer.add_image('spatial distribution', sum_dist, epoch, dataformats='HW')
        writer.add_images('samples', sample[0:64,:,:,:], epoch)
        '''

def save_ckpt(epoch, net, optimizer, dir_name):
    torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/' + dir_name[:])
    #np.save('samples/' + dir_name[:] + '_epoch=%d' % epoch, sample)

def load_all_pickles(path):
    outputs = []
    print('loading all .pkl files from',path)
    files = [ f for f in listdir(path) if isfile(join(path,f)) ]
    for f in files:
        if f[-4:] in ('.pkl'):
            name = f[:-4]+'_'+f[-3:]
            print('loading', f, 'as', name)
            with open(path + '/' + f, 'rb') as f:
                outputs.append(pickle.load(f))

    return outputs

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def find_dir():
    found = 0
    ii = 0
    while found == 0:
        ii += 1
        if not os.path.exists('logfiles/run_%d' % ii):
            found = 1

    return ii
