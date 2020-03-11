import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedConv2d(nn.Conv2d):  # add a mask to the regular Conv2D function, so that it cannot learn about the future
    def __init__(self, mask_type, channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        # spatial masking - prevent information from neighbours
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0  # mask type B allows access to the 'present' pixel, mask A does not
        self.mask[:, :, kH // 2 + 1:] = 0

        if channels > 1:
            # channel masking - block information from nearby color channels - ONLY 2 CHANNELS
            ''' 
            filters will be stacked as x1,x2,x3,x1,x2,x3,... , therefore, we will mask such that 
            e.g. filter 2 serving x2 can see previous outputs from x3, but not x1
            we will achieve this by building a connections graph, which will zero-out all elements from given channels 
            '''
            # mask A only allows information from lower channels
            Cin = self.mask.shape[1] # input filters
            Cout = self.mask.shape[0] # output filters
            def channel_mask(i_out, i_in): # a map which defines who is allowed to see what
                cout_idx = np.expand_dims(np.arange(Cout) % 2 == i_out, 1)
                cin_idx = np.expand_dims(np.arange(Cin) % 2 == i_in, 0)
                a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
                return a1 * a2

            mask = np.array(self.mask)
            for c in range(2): # mask B allows information from current and lower channels
                mask[channel_mask(c, c), kH // 2, kW // 2] = 0.0 if mask_type == 'A' else 1.0

            mask[channel_mask(0, 1), kH // 2, kW // 2] = 0.0
            self.mask = torch.from_numpy(mask)
    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d, self).forward(x)

class DoubleMaskedConv2d(nn.Conv2d):  # adds to regular masked conv2d by masking also the input in subsequent layers (densenet only)
    def __init__(self, mask_type, *args, **kwargs):
        super(DoubleMaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, self.kH, self.kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks
        self.mask[:, :, self.kH // 2, self.kW // 2 + (mask_type == 'B'):] = 0  # mask type B allows access to the 'present' pixel, mask A does not
        self.mask[:, :, self.kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        self.weight.data[0,:, self.kH//2, self.kW//2] *=0 # mask the central pixel of the first filter (which will always be the input in a densent)
        return super(DoubleMaskedConv2d, self).forward(x)


class MaskedPointwiseConv2d(nn.Conv2d):  # adds to regular masked conv2d by masking also the input in subsequent layers (densenet only)
    def __init__(self, *args, **kwargs):
        super(MaskedPointwiseConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data[:,0, 0, 0] *=0 # mask the entirety of the first filter (which will always be the input in a densenet)
        return super(MaskedPointwiseConv2d, self).forward(x)


def gated_activation(input):
    # implement gated activation from Conditional Generation with PixelCNN Encoders
    assert (input.shape[1] % 2) == 0
    a, b = torch.chunk(input, 2, 1) # split input into two equal parts - only works for even number of filters
    a = F.tanh(a)
    b = F.sigmoid(b)

    return torch.matmul(a,b) # return element-wise (sigmoid-gated) product


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return gated_activation(input)


class Activation(nn.Module):
    def __init__(self, activation_func, *args, **kwargs):
        super().__init__()
        if activation_func == 'gated':
            self.activation = gated_activation
        elif activation_func == 'relu':
            self.activation = F.relu

    def forward(self, input):
        return self.activation(input)


class StackedConvolution(nn.Module):
    def __init__(self, f_in, f_out, padding, dilation, *args, **kwargs):
        super(StackedConvolution, self).__init__(*args, **kwargs)

        self.v_BN = nn.BatchNorm2d(f_in)
        self.v_Conv2d = nn.Conv2d(f_in, 2 * f_out, (2, 3), 1, padding, dilation, bias=True, padding_mode='zeros')
        self.v_to_h_fc = nn.Conv2d(2 * f_out, f_out, 1)
        self.h_BN = nn.BatchNorm2d(f_in)
        self.h_Conv2d = nn.Conv2d(f_in, f_out, (1, 2), 1, padding, dilation, bias=True, padding_mode='zeros')
        self.h_to_h = nn.Conv2d(f_out, f_out, 1)
        self.activation = Activation('gated') # for ReLU, must change number of filters as gated approach halves filters on each application

    def forward(self, v_in, h_in):
        residue = h_in * 1 # residual track

        v_in = self.v_Conv2d(self.v_BN(v_in)) # vertical stack
        v_out = self.activation(v_in)
        v_to_h = self.v_to_h_fc(v_in)

        h_in = self.h_Conv2d(self.h_BN(v_in))
        h_out = self.activation(torch.cat((h_in,v_to_h[:, :, 1: , :]),1))
        h_out = self.h_to_h(h_out) + residue

        return v_out, h_out


class PixelCNN(nn.Module):
    def __init__(self, filters, filter_size, layers, out_maps, padded):
        super(PixelCNN, self).__init__()

        if padded == 1:
            padding = (filter_size - 1) //2
        else:
            padding = 0

        self.initial_convolution = MaskedConv2d('A', 1, filters, filter_size, 1, (filter_size -1) // 2, padding_mode = 'zeros', bias = True)
        self.initial_batch_norm = nn.BatchNorm2d(filters)
        self.hidden_convolutions = nn.ModuleList([MaskedConv2d('B', filters, filters, filter_size, 1, padding, padding_mode = 'zeros', bias = True) for i in range(layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.output_convolution = nn.Conv2d(filters, out_maps, 1)

    def forward(self, x):
        x = F.relu(self.initial_batch_norm(self.initial_convolution(x)))
        for i in range(len(self.hidden_convolutions)):
            x = F.relu(self.batch_norms[i](self.hidden_convolutions[i](x)))
        x = self.output_convolution(x)
        return x


class PixelCNN_RES(nn.Module):
    def __init__(self, filters, filter_size, layers, out_maps, channels, padded):
        super(PixelCNN_RES, self).__init__()

        if padded == 1:
            padding = 1
        else:
            padding = 0

        self.initial_batch_norm = nn.BatchNorm2d(channels)#2 * filters)
        self.initial_convolution = MaskedConv2d('A', channels, channels, 2*filters, filter_size, 1, (filter_size - 1) // 2, padding_mode='zeros', bias=True)
        self.hidden_convolutions = nn.ModuleList([MaskedConv2d('B', channels, filters, filters, 3, 1, padding, padding_mode='zeros', bias=True) for i in range(layers)])
        self.shrink_features = nn.ModuleList([nn.Conv2d(2 * filters, filters, 1) for i in range(layers)])
        self.grow_features = nn.ModuleList([nn.Conv2d(filters, 2 * filters, 1) for i in range(layers)])
        self.batch_norms_1 = nn.ModuleList([nn.BatchNorm2d(2 * filters) for i in range(layers)])
        self.batch_norms_2 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.batch_norms_3 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.fc1 = nn.Conv2d(2 * filters, 256, 1)
        self.fc2 = nn.Conv2d(256, out_maps * channels, 1)

    def forward(self, x): # pre-activated residual model
        x = F.relu(self.initial_convolution(self.initial_batch_norm(x)))

        for i in range(len(self.hidden_convolutions)):
            residue = x
            x = self.shrink_features[i](F.relu(self.batch_norms_1[i](x)))
            x = self.hidden_convolutions[i](F.relu(self.batch_norms_2[i](x)))
            x = self.grow_features[i](F.relu(self.batch_norms_3[i](x)))
            x += residue

        x = self.fc2(F.relu(self.fc1(x)))
        return x


class PixelCNN_RES_OUT(nn.Module):
    def __init__(self, filters, filter_size, layers, out_maps, channels, padded):
        super(PixelCNN_RES_OUT, self).__init__()

        if padded == 1:
            padding = 1
        else:
            padding = 0

        self.initial_batch_norm = nn.BatchNorm2d(channels)  # 2 * filters)
        self.initial_convolution = MaskedConv2d('A', channels, channels, 2 * filters, filter_size, 1, padding, bias=True)
        self.hidden_convolutions = nn.ModuleList([MaskedConv2d('B', channels, filters, filters, 3, 1, padding, bias=True) for i in range(layers)])
        self.shrink_features = nn.ModuleList([nn.Conv2d(2 * filters, filters, 1) for i in range(layers)])
        self.grow_features = nn.ModuleList([nn.Conv2d(filters, 2 * filters, 1) for i in range(layers)])
        self.batch_norms_1 = nn.ModuleList([nn.BatchNorm2d(2 * filters) for i in range(layers)])
        self.batch_norms_2 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.batch_norms_3 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.fc1 = nn.Conv2d(2 * filters, 256, 1)
        self.fc2 = nn.Conv2d(256, out_maps * channels, 1)

    def forward(self, x): # pre-activated residual model
        x = F.relu(self.initial_convolution(self.initial_batch_norm(x)))

        for i in range(len(self.hidden_convolutions)):
            residue = x
            x = self.shrink_features[i](F.relu(self.batch_norms_1[i](x)))
            x = self.hidden_convolutions[i](F.relu(self.batch_norms_2[i](x)))
            x = self.grow_features[i](F.relu(self.batch_norms_3[i](x)))
            x += residue[:,:,1:-1,1:-1] #contract input

        x = self.fc2(F.relu(self.fc1(x)))
        return x


class PixelDRN(nn.Module):  # dilated residual network
    def __init__(self, filters, initial_filter_size, dilation, layers, out_maps, grow_filters, padding):
        super(PixelDRN, self).__init__()

        if grow_filters == 1:
            growth_factor = 1  # the (exponential) rate at which filters grow from block to block
        elif grow_filters == 0:
            growth_factor = 0

        bottleneck_factor = 2  # the grow/shrink ratio of our 1x1 convolutions)
        bottleneck_filters = [int(filters * 2 ** (torch.arange(7, dtype=torch.float32) * growth_factor)[i]) for i in range(7)]  # we will grow filters in blocks of resnet layers, maxing out in block D
        bottleneck_filters[4], bottleneck_filters[5], bottleneck_filters[6] = [bottleneck_filters[3], bottleneck_filters[3], bottleneck_filters[3]]
        self.block_dilation = [int(dilation ** torch.tensor([0, 0, 0, 1, 2, 1, 0],dtype=torch.int32)[i]) for i in range(7)]  # the dilation amount for each block - coincidentally also the unpadding amount
        self.keep_residues = [1,1,1,1,0,0] # toss residues for last two layers
        #self.keep_residues = [1, 1, 1, 1, 1, 1]  # toss residues for last two layers
        self.unpadding = self.block_dilation * (1 - padding)  # how much we need to unpad residues in generation

        self.batch_norms_1,self.shrink_features,self.batch_norms_2,self.convolution,self.batch_norms_3,self.grow_features,self.grow_residues = [[],[],[],[],[],[],[]]

        # stem
        self.initial_convolution = MaskedConv2d('A', 1, bottleneck_filters[0], initial_filter_size, 1, padding * (initial_filter_size - 1) // 2, padding_mode='zeros', bias=True)
        self.initial_batch_norm = nn.BatchNorm2d(bottleneck_filters[0])
        self.initial_grow_features = nn.Conv2d(bottleneck_filters[0], bottleneck_factor * bottleneck_filters[0], 1)

        # blocks A - F
        for ii in range(1,7):
            self.batch_norms_1_A = nn.ModuleList([nn.BatchNorm2d(bottleneck_factor * bottleneck_filters[ii]) for i in range(layers)])
            self.shrink_features_A = nn.ModuleList([nn.Conv2d(bottleneck_factor * bottleneck_filters[ii], bottleneck_filters[ii], 1) for i in range(layers)])
            self.batch_norms_1_A[0] = nn.BatchNorm2d(bottleneck_factor * bottleneck_filters[ii - 1])  # must comport with previous block filters
            self.shrink_features_A[0] = nn.Conv2d(bottleneck_factor * bottleneck_filters[ii - 1], bottleneck_filters[ii], 1)
            
            self.batch_norms_1.append(self.batch_norms_1_A)
            self.shrink_features.append(self.shrink_features_A)
            self.batch_norms_2.append(nn.ModuleList([nn.BatchNorm2d(bottleneck_filters[ii]) for i in range(layers)]))
            self.convolution.append(nn.ModuleList([MaskedConv2d('B', bottleneck_filters[ii], bottleneck_filters[ii], 3, 1, padding * self.block_dilation[ii], self.block_dilation[ii], padding_mode='zeros', bias=True) for i in range(layers)]))
            self.batch_norms_3.append(nn.ModuleList([nn.BatchNorm2d(bottleneck_filters[ii]) for i in range(layers)]))
            self.grow_features.append(nn.ModuleList([nn.Conv2d(bottleneck_filters[ii], bottleneck_factor * bottleneck_filters[ii], 1) for i in range(layers)]))
    
            self.grow_residues.append(nn.Conv2d(bottleneck_factor * bottleneck_filters[ii-1], bottleneck_factor * bottleneck_filters[ii], 1))
            
        # output
        self.output_convolution = nn.Conv2d(bottleneck_factor * bottleneck_filters[-1], out_maps, 1)

        # to make pytorch happy
        self.batch_norms_1 = nn.ModuleList(self.batch_norms_1)
        self.batch_norms_2 = nn.ModuleList(self.batch_norms_2)
        self.batch_norms_3 = nn.ModuleList(self.batch_norms_3)
        self.shrink_features = nn.ModuleList(self.shrink_features)
        self.grow_features = nn.ModuleList(self.grow_features)
        self.convolution = nn.ModuleList(self.convolution)
        self.grow_residues = nn.ModuleList(self.grow_residues)

    def forward(self, x): # pre-activated residual model
        x = F.relu(self.initial_grow_features(self.initial_batch_norm(self.initial_convolution(x))))

        for i in range(len(self.convolution)):
            for j in range(len(self.convolution[i])):
                residue = x * 1
                x = self.shrink_features[i][j](F.relu(self.batch_norms_1[i][j](x)))
                x = self.convolution[i][j](F.relu(self.batch_norms_2[i][j](x)))
                x = self.grow_features[i][j](F.relu(self.batch_norms_3[i][j](x)))

                if self.keep_residues[i]:
                    if j == 0:  # on the first layer of a new block we need to boost residue size, if we are keeping them at all
                        residue = self.grow_residues[i](residue)
                if self.unpadding != []:
                    x += residue[:, :, self.unpadding[i + 1]:-self.unpadding[i + 1], self.unpadding[i + 1]:-self.unpadding[i + 1]] * self.keep_residues[i]  # keep residues of the same size as the output, and only when desired
                else:
                    x += residue * self.keep_residues[i]
        
        x = self.output_convolution(x)
        return x


class DensePixelDCNN(nn.Module):  # (ungated) Dense PixelCNN with dilated (soon) convolutions
    def __init__(self, filters, initial_convolution_size, dilation, layers, out_maps, padding):
        super(DensePixelDCNN, self).__init__()
        # #filters=k --> is the growth rate f = k0+k*(layers-1)

        self.padding = padding
        #self.block_dilation = [int(dilation ** torch.tensor([0, 0, 0, 1, 2, 1, 0], dtype=torch.int32)[i]) for i in range(7)]  # the dilation amount for each block - coincidentally also the unpadding amount
        if self.padding == 0:
            self.unpadding = (np.ones(layers + 1)* (1 - padding)).astype('uint8') # how much we need to unpad shortcuts in generation
            self.unpadding[0] = (initial_convolution_size-1)//2
        else:
            self.unpadding = np.zeros(layers + 1).astype('uint8')

        self.cumulative_unpad = np.zeros((layers+1,layers+1))
        for i in range(layers+1):
            for j in range(i):
                self.cumulative_unpad[i,j] = np.sum(self.unpadding[j:i])

        self.input_depth = 1
        initial_filters = filters * 2
        self.filters_in = self.input_depth + filters * (np.arange(layers)) + initial_filters # 1 here is the number of input feature maps

        # stem
        self.initial_convolution = DoubleMaskedConv2d('A', self.input_depth, initial_filters, initial_convolution_size, 1, padding * (initial_convolution_size - 1) // 2, padding_mode='zeros', bias=True)
        self.initial_batch_norm = nn.BatchNorm2d(self.input_depth)
        self.output_convolution = nn.Conv2d(filters, out_maps, 1)

        # conv layers for a block (blocks tbd)
        self.batch_norm, self.convolution = [[], []]
        blocks = 1
        for ii in range(blocks):
            self.batch_norm = nn.ModuleList([nn.BatchNorm2d(self.filters_in[i]) for i in range(layers)])
            self.convolution = nn.ModuleList([DoubleMaskedConv2d('B', self.filters_in[i], filters, 3, 1, padding, 1, padding_mode='zeros', bias=True) for i in range(layers)])
            self.batch_norm2 = nn.ModuleList([nn.BatchNorm2d(self.filters_in[i]) for i in range(layers)])
            self.pointwise = nn.ModuleList([MaskedPointwiseConv2d(self.filters_in[i], self.filters_in[i], 1) for i in range(layers)])

        # to make pytorch happy
        self.batch_norm = nn.ModuleList(self.batch_norm)
        self.convolution = nn.ModuleList(self.convolution)

    def forward(self, x):  # densenet

        self.residues = []
        masked_input = x
        self.residues.append(masked_input)
        self.residues.append(F.relu(self.initial_convolution(self.initial_batch_norm(x)))) #stem

        for i in range(2,len(self.convolution)+2): # dense layers in a block

            #self.residues.append(self.convolution[i - 2](F.relu(self.batch_norm[i - 2](torch.cat([self.residues[j][:,:,int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-2]-int(self.cumulative_unpad[i-1,j]), int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-1]-int(self.cumulative_unpad[i-1,j])] for j in range(len(self.residues))], 1)))))
            self.residues.append(self.convolution[i - 2](F.relu(self.batch_norm[i - 2](F.relu(self.pointwise[i - 2](self.batch_norm2[i - 2](torch.cat([self.residues[j][:,:,int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-2]-int(self.cumulative_unpad[i-1,j]), int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-1]-int(self.cumulative_unpad[i-1,j])] for j in range(len(self.residues))], 1))))))))


        return self.output_convolution(self.residues[-1])


class PixelCNN2(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, initial_convolution_size, dilation, blocks, out_maps, padding):
        super(PixelCNN2, self).__init__()

        ### initialize constants
        self.padding = padding
        self.layers_per_block = 1
        self.blocks = blocks
        self.layers = int(self.layers_per_block * blocks)
        initial_filters = filters
        self.input_depth = 1 # input channels
        f_in = (np.ones(self.layers + 1) * filters).astype(int)
        f_out = (np.ones(self.layers + 1) * filters).astype(int)
        self.dilation = (np.ones(self.layers + 1) * dilation).astype(int)

        if self.padding == 0:
            self.unpad = np.zeros(self.layers + 1).astype(int)
            for i in range(1,self.layers):
                self.unpad[i] = dilation[i].astype(int)

            self.unpad [0] = (initial_convolution_size-1)//2
        else:
            self.unpad = np.zeros(self.layers + 1).astype(int)
        ###

        # initial layer
        self.initial_batch_norm = nn.BatchNorm2d(self.input_depth)
        self.v_initial_convolution = MaskedConv2d('A', self.input_depth, 2 * initial_filters, (initial_convolution_size//2 + 1, initial_convolution_size), 1, padding * (initial_convolution_size - 1) // 2, padding_mode='zeros', bias=True)
        self.v_to_h_initial = nn.Conv2d(2 * initial_filters, initial_filters, 1)
        self.h_initial_convolution = MaskedConv2d('A', self.input_depth, initial_filters, (1, initial_convolution_size//2), 1, padding * (initial_convolution_size - 1) // 2, padding_mode='zeros', bias=True)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1)

        #output layers
        self.fc1 = Activation(nn.Conv2d(f_out[-1], 256, 1))
        self.fc2 = Activation(nn.Conv2d(256, out_maps, 1))

        # stack layers in blocks
        self.conv_layer = []
        for j in range(blocks):
            self.conv_layer.append([StackedConvolution(f_in[i + j * self.layers_per_block], f_out[i + j * self.layers_per_block], padding, self.dilation[i + j * self.layers_per_block]) for i in range(self.layers_per_block)]) # stacked convolution (no blind spot)


    def forward(self, input):
        # initial convolution
        input = self.initial_batch_norm(input)
        # separate stacks
        v_data = self.v_initial_convolution(input)
        v_to_h_data = self.v_to_h_initial(v_data)
        h_data = self.h_initial_convolution(input)
        h_data = Activation(torch.cat((v_to_h_data[:,:,1:,:], h_data), dim=1))
        h_data = self.h_to_h_initial(h_data)
        v_data = Activation(v_data)

        for i in range(self.blocks): # loop over convolutional layers
            for j in range(self.layers_per_block):
               v_data, h_data = self.conv_layer[i][j](v_data, h_data) # stacked convolutions fix blind spot

        # output convolutions
        x = self.activation(self.fc1(h_data))
        x = self.activation(self.fc2(x))

        return x
