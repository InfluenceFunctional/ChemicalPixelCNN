import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedConv2d(nn.Conv2d):  # add a mask to the regular Conv2D function, so that it cannot learn about the future
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0  # mask type B allows access to the 'present' pixel, mask A does not
        self.mask[:, :, kH // 2 + 1:] = 0

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
        self.weight.data[:,0, self.kH//2, self.kW//2] *=0 # mask the central pixel of the first filter (which will always be the input in a densent)
        return super(DoubleMaskedConv2d, self).forward(x)

class MaskedPointwiseConv2d(nn.Conv2d):  # adds to regular masked conv2d by masking also the input in subsequent layers (densenet only)
    def __init__(self, *args, **kwargs):
        super(MaskedPointwiseConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data[:,0, 0, 0] *=0 # mask the entirety of the first filter (which will always be the input in a densenet)
        return super(MaskedPointwiseConv2d, self).forward(x)

class GatedActivation():
    def __init__(selfself, *args, **kwargs):

class StackedConvolution(nn.Module):
    def __init__(self, f_in, f_out, padding, dilation, *args, **kwargs):
        super(StackedConvolution, self).__init(*args, **kwargs)

        self.v_BN = nn.BatchNorm2d(f_in)
        self.v_Conv2d = nn.Conv2d(f_in, f_out, (2, 3), 1, padding, dilation, bias=True, padding_mode='zeros')
        self.v_to_h_fc = nn.Conv2d(f_in, f_out, 1)
        self.h_BN = nn.BatchNorm2d(f_in, f_out)
        self.h_Conv2d = nn.Conv2d(f_in, f_out, (1, 2), 1, padding, dilation, bias=True, padding_mode='zeros')
        self.h_to_h = nn.Conv2d(f_in, f_out, 1)


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
    def __init__(self, filters, filter_size, layers, out_maps, padded):
        super(PixelCNN_RES, self).__init__()

        if padded == 1:
            padding = 1
        else:
            padding = 0

        self.initial_convolution = MaskedConv2d('A', 1, 2*filters, filter_size, 1, (filter_size - 1) // 2, padding_mode='zeros', bias=True)
        self.initial_batch_norm = nn.BatchNorm2d(1)#2 * filters)
        self.hidden_convolutions = nn.ModuleList([MaskedConv2d('B', filters, filters, 3, 1, padding, padding_mode='zeros', bias=True) for i in range(layers)])
        self.shrink_features = nn.ModuleList([nn.Conv2d(2 * filters, filters, 1) for i in range(layers)])
        self.grow_features = nn.ModuleList([nn.Conv2d(filters, 2 * filters, 1) for i in range(layers)])
        self.batch_norms_1 = nn.ModuleList([nn.BatchNorm2d(2 * filters) for i in range(layers)])
        self.batch_norms_2 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.batch_norms_3 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.output_convolution = nn.Conv2d(2 * filters, out_maps, 1)

    def forward(self, x): # pre-activated residual model
        x = F.leaky_relu(self.initial_convolution(self.initial_batch_norm(x)))

        for i in range(len(self.hidden_convolutions)):
            residue = x
            x = self.shrink_features[i](F.leaky_relu(self.batch_norms_1[i](x)))
            x = self.hidden_convolutions[i](F.leaky_relu(self.batch_norms_2[i](x)))
            x = self.grow_features[i](F.leaky_relu(self.batch_norms_3[i](x)))
            x += residue

        x = self.output_convolution(x)
        return x

class PixelCNN_RES_OUT(nn.Module):
    def __init__(self, filters, filter_size, layers, out_maps, padded):
        super(PixelCNN_RES_OUT, self).__init__()

        if padded == 1:
            padding = 1
        else:
            padding = 0

        self.initial_convolution = MaskedConv2d('A', 1, 2*filters, filter_size, 1, padding, padding_mode='zeros', bias=True)
        self.initial_batch_norm = nn.BatchNorm2d(1)#2 * filters)
        self.hidden_convolutions = nn.ModuleList([MaskedConv2d('B', filters, filters, 3, 1, padding, padding_mode='zeros', bias=True) for i in range(layers)])
        self.shrink_features = nn.ModuleList([nn.Conv2d(2 * filters, filters, 1) for i in range(layers)])
        self.grow_features = nn.ModuleList([nn.Conv2d(filters, 2 * filters, 1) for i in range(layers)])
        self.batch_norms_1 = nn.ModuleList([nn.BatchNorm2d(2 * filters) for i in range(layers)])
        self.batch_norms_2 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.batch_norms_3 = nn.ModuleList([nn.BatchNorm2d(filters) for i in range(layers)])
        self.output_convolution = nn.Conv2d(2 * filters, out_maps, 1)

    def forward(self, x): # pre-activated residual model
        x = F.leaky_relu(self.initial_convolution(self.initial_batch_norm(x)))

        for i in range(len(self.hidden_convolutions)):
            residue = x
            x = self.shrink_features[i](F.leaky_relu(self.batch_norms_1[i](x)))
            x = self.hidden_convolutions[i](F.leaky_relu(self.batch_norms_2[i](x)))
            x = self.grow_features[i](F.leaky_relu(self.batch_norms_3[i](x)))
            x += residue[:,:,1:-1,1:-1] #contract input

        x = self.output_convolution(x)
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
        initial_filters = filters
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
        self.residues.append(self.initial_convolution(F.relu(self.initial_batch_norm(x)))) #stem

        for i in range(2,len(self.convolution)+2): # dense layers in a block

            #self.residues.append(self.convolution[i - 2](F.leaky_relu(self.batch_norm[i - 2](torch.cat([self.residues[j][:,:,int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-2]-int(self.cumulative_unpad[i-1,j]), int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-1]-int(self.cumulative_unpad[i-1,j])] for j in range(len(self.residues))], 1)))))
            self.residues.append(self.convolution[i - 2](F.relu(self.batch_norm[i - 2](F.relu(self.pointwise[i - 2](self.batch_norm2[i - 2](torch.cat([self.residues[j][:,:,int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-2]-int(self.cumulative_unpad[i-1,j]), int(self.cumulative_unpad[i-1,j]):self.residues[j].shape[-1]-int(self.cumulative_unpad[i-1,j])] for j in range(len(self.residues))], 1))))))))


        return self.output_convolution(self.residues[-1])

'''
class PixelCNN2(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, filters_per_block, initial_convolution_size, dilation, blocks, out_maps, padding):
        super(PixelCNN2, self).__init__()
        #initialize activation
        activation = 'ReLU'
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        #elif activation == 'gated':
            #self.activation = torch.matmul(torch.chunk())

        # initialize constants
        self.padding = padding
        self.layers_per_block = 5
        self.blocks = blocks
        self.layers = self.layers_per_block * blocks
        initial_filters = filters
        self.input_depth = 1 # input channels
        f_in = np.ones(self.layers + 1) * filters
        f_out = np.ones(self.layers + 1) * filters
        dilation = np.ones(self.layers + 1)

        if self.padding == 0:
            self.unpad = np.zeros(self.layers + 1)
            for i in range(1,self.layers):
                self.unpad[i] = dilation[i].astype('uint8')

            self.unpad [0] = (initial_convolution_size-1)//2
        else:
            self.unpad = np.zeros(self.layers + 1).astype('uint8')

        # initial layer
        self.initial_batch_norm = nn.BatchNorm2d(self.input_depth)
        self.v_initial_convolution = MaskedConv2d('A', self.input_depth, initial_filters, (initial_convolution_size//2 + 1, initial_convolution_size), 1, padding * (initial_convolution_size - 1) // 2, padding_mode='zeros', bias=True)
        self._initial_convolution = MaskedConv2d('A', self.input_depth, initial_filters, (1, initial_convolution_size//2), 1, padding * (initial_convolution_size - 1) // 2, padding_mode='zeros', bias=True)

        #output layers
        self.fc1 = nn.Conv2d(f_out[-1], 256, 1)
        self.fc2 = nn.Conv2d(256, out_maps, 1)

        # initalize blocks
        self.v_BN,self.v_Conv2d,self.v_to_h_fc,self.h_BN,self.h_Conv2d,self.h_to_h = [[],[],[],[],[],[]]

        # stack layers in blocks
        for ii in range(blocks):
            self.v_BN.append(nn.ModuleList([nn.BatchNorm2d(self.f_in[i]) for i in range(self.layers)]))
            self.v_Conv2d.append(nn.ModuleList([nn.Conv2d(self.f_in[i], self.f_out[i], (2, 3), 1, self.padding, self.dilation[ii], bias=True, padding_mode='zeros') for i in range(self.layers)]))
            self.v_to_h_fc.append(nn.ModuleList([nn.Conv2d(self.f_out[i], self.f_out[i], 1) for i in range(self.layers)]))
            self.h_BN.append(nn.ModuleList([nn.BatchNorm2d(self.f_in[i], self.f_out[i]) for i in range(self.layers)]))
            self.h_Conv2d.append(nn.ModuleList([nn.Conv2d(self.f_in[i], self.f_out[i], (1, 2), 1, self.padding, self.dilation[ii], bias=True, padding_mode='zeros') for i in range(self.layers)]))
            self.h_to_h.append(nn.ModuleList([nn.Conv2d(self.f_in[i], self.f_out[i], 1) for i in range(self.layers)]))

    def forward(self, x):
        # initial convolutions
        # v batch norm
        # v activation
        # v convolution
        # v to h

        # h batch norm
        # h activation
        # h convolution (incl v)

        # skips, residuals and outputs

        for i in self.blocks:
            for j in self.layers_per_block:
                # v batch norm
                # v activation
                # v convolution
                # v to h

                # h batch norm
                # h activation
                # h convolution (incl v)

                # skips, residuals and outputs

        # output convolutions
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        return x
'''