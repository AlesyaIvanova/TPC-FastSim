# import tensorflow as tf
import torch
from torch.autograd import Variable

# DO NOT REMOVE. used by code inside eval()
import numpy as np  # noqa: F401

custom_objects = {}

activations = {
    'NULL': None,
    'elu': torch.nn.ELU(),
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh(),
}

def get_activation(activation_name):
    try:
        activation = activations[activation_name]
    except KeyError:
        activation = eval(activation_name)
    return activation


class FullyConnectedBlock(torch.nn.Module):
    
    def __init__(
        self,
        units, 
        activations, 
        input_shape=None, 
        output_shape=None, 
        dropouts=None, 
        name=None,
        kernel_init=None,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        assert len(units) == len(activations)
        if dropouts:
            assert len(dropouts) == len(units)
    
        activations = [get_activation(a) for a in activations]
        self.layers = torch.nn.ParameterList()
    
        print(units)
        for i in range(len(units)):
            if i == 0 and input_shape:
                self.layers.append(torch.nn.Linear(in_features=input_shape[0], out_features=units[i]))
            else:
                self.layers.append(torch.nn.Linear(in_features=units[i-1], out_features=units[i]))
            
            if activations[i] is not None:
                self.layers.append(activations[i])
    
            if dropouts and dropouts[i]:
                self.layers.append(torch.nn.Dropout(p=dropouts[i]))
         
                
    def forward(self, input_tensor) -> Variable:
        x = input_tensor
        # print('start')
        for layer in self.layers:
            # print(x.shape)
            # print(layer)
            x = layer(x)
        
        if self.output_shape:
            torch.reshape(x, shape=(-1, *self.output_shape))
        return x


class SingleBlock(torch.nn.Module):
    
    def __init__(
        self,
        input_shape, 
        output_shape, 
        activation=None, 
        batchnorm=None, 
        dropout=None
    ) -> None:
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.linear_layer = torch.nn.Linear(in_features=input_shape, output_features=output_shape)
        if batchnorm:
            self.batchnorm_layer = torch.nn.BatchNorm2d()
        if dropout:
            self.dropout_layer = torch.nn.Dropout(p=dropout)
            
    
    def forward(self, input_tensor) -> Variable:
        x = self.linear(input_tensor)
        if self.batchnorm:
            x = self.batchnorm_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout_layer(x)
        return x


class FullyConnectedResidualBlock(torch.nn.Module):
    
    def __init__(
        self,
        units,
        activations,
        input_shape,
        kernel_init='glorot_uniform',
        batchnorm=True,
        output_shape=None,
        dropouts=None,
        name=None,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        assert isinstance(units, int)
        if dropouts:
            assert len(dropouts) == len(activations)
        else:
            dropouts = [None] * len(activations)
    
        activations = [get_activation(a) for a in activations]
        self.blocks = torch.nn.ParameterList()
        
        for i, (act, dropout) in enumerate(zip(activations, dropouts)):
            self.blocks.append(SingleBlock(
                input_shape=(units if i > 0 else input_shape), 
                output_shape=units, 
                activations=act, 
                batchnorm=batchnorm, 
                dropout=dropout,
            ))


    def forward(self, input_tensor) -> Variable: 
        x = input_tensor
        for i, single_block in enumerate(self.blocks):
            if len(x.shape) == 2 and x.shape[1] == units:
                x = x + single_block(x)
            else:
                assert i == 0
                x = single_block(x)
    
        if self.output_shape:
            x = torch.reshape(x, self.output_shape)
        return x


class ConcatBlock(torch.nn.Module):
    
    def __init__(
        self,
        input1_shape, 
        input2_shape, 
        reshape_input1=None, 
        reshape_input2=None, 
        dim=-1, 
        name=None
    ) -> None:
        super().__init__()
        self.input1_shape = input1_shape
        self.input1_shape = input1_shape
        self.reshape_input1 = reshape_input1
        self.reshape_input2 = reshape_input2
        self.dim = dim
        
        
    def forward(self, in1, in2) -> Variable:
        concat1, concat2 = in1, in2
        if self.reshape_input1:
            concat1 = torch.reshape(concat1, self.reshape_input1)
        if self.reshape_input2:
            concat2 = torch.reshape(concat2, self.reshape_input2)
        out = torch.cat((concat1, concat2), dim=self.dim)
        return out


class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        filters,
        kernel_sizes,
        paddings,
        activations,
        poolings,
        input_shape,
        output_shape=None,
        kernel_init='glorot_uniform',
        dropouts=None,
        name=None,
    ) -> None:
        super().__init__()
        assert len(filters) == len(kernel_sizes) == len(paddings) == len(activations) == len(poolings)
        if dropouts:
            assert len(dropouts) == len(filters)

        self.output_shape = output_shape
        activations = [get_activation(a) for a in activations]
        self.layers = torch.nn.ParameterList()
        
        # print(input_shape, filters)
        for i, (nfilt, ksize, padding, act, pool) in enumerate(zip(filters, kernel_sizes, paddings, activations, poolings)):
            self.layers.append(torch.nn.Conv2d(
                in_channels=(input_shape if i == 0 else filters[i-1]),
                out_channels=nfilt,
                kernel_size=ksize,
                padding=padding,
            ))
            
            if act is not None:
                self.layers.append(act)
    
            if dropouts and dropouts[i]:
                self.layers.append(torch.nn.Dropout(p=dropouts[i]))
    
            if pool:
                self.layers.append(torch.nn.MaxPool2d(kernel_size=pool))
                
                
    def forward(self, input_tensor) -> Variable: 
        x = input_tensor
        for layer in self.layers:
            # print('a', x.shape)
            # print('a', layer)
            x = layer(x)
            
        if self.output_shape:
            x = torch.reshape(x, (-1, self.output_shape))
        return x


class VectorImgConnectBlock(torch.nn.Module):
    
    def __init__(
        self,
        vector_shape, 
        img_shape, 
        block, 
        vector_bypass=False, 
        concat_outputs=True, 
        name=None
    ) -> None:
        super().__init__()
        self.vector_shape = tuple(vector_shape)
        self.img_shape = tuple(img_shape)
        self.block = block
        self.vector_bypass = vector_bypass
        self.concat_outputs = concat_outputs

        assert len(vector_shape) == 1
        assert 2 <= len(img_shape) <= 3
    
    
    def forward(self, input_tensors) -> Variable:
        input_vec, input_img = input_tensors[0], input_tensors[1]
        # print(input_vec.shape, input_img.shape)
        block_input = input_img
        if len(self.img_shape) == 2:
            block_input = torch.reshape(block_input, (-1, *(self.img_shape + (1,))))
        if not self.vector_bypass:
            reshaped_vec = torch.tile(torch.reshape(input_vec, (-1, *((1, 1) + self.vector_shape))), dims=(1, *self.img_shape[:2], 1))
            block_input = torch.cat((block_input, reshaped_vec), dim=-1)

        # print('block', self.block)
        block_input = torch.permute(block_input, (0, 3, 1, 2))
        # print('n', block_input.shape)
        block_output = self.block(block_input)
    
        outputs = [input_vec, block_output]
        # print('d', input_vec.shape, block_output.shape)
        if self.concat_outputs:
            outputs = torch.cat(outputs, dim=-1)
        return outputs


def build_block(block_type, arguments):
    if block_type == 'fully_connected':
        block = FullyConnectedBlock(**arguments)
    elif block_type == 'conv':
        block = ConvBlock(**arguments)
    elif block_type == 'connect':
        inner_block = build_block(**arguments['block'])
        arguments['block'] = inner_block
        block = VectorImgConnectBlock(**arguments)
    elif block_type == 'concat':
        block = ConcatBlock(**arguments)
    elif block_type == 'fully_connected_residual':
        block = FullyConnectedResidualBlock(**arguments)
    else:
        raise (NotImplementedError(block_type))

    return block


class FullModel(torch.nn.Module):
    
    def __init__(
        self,
        block_descriptions,
        name=None, 
        custom_objects_code=None,
    ) -> None:
        super().__init__()
        if custom_objects_code:
            print("build_architecture(): got custom objects code, executing:")
            print(custom_objects_code)
            exec(custom_objects_code, globals(), custom_objects)
    
        print(block_descriptions)
        self.blocks = torch.nn.ParameterList(build_block(**descr) for descr in block_descriptions)
    
    
    def forward(self, inputs) -> Variable:
        outputs = inputs
        for block in self.blocks:
            # print('q', len(outputs))
            # for output in outputs:
              # print('qq', output.shape)
            # print('qqq', block)
            outputs = block(outputs)
            # print('aaaa', outputs.grad)
        return outputs
