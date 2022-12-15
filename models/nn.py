# import tensorflow as tf
import torch

# DO NOT REMOVE. used by code inside eval()
import numpy as np  # noqa: F401

custom_objects = {}

activations = {
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh(),
}

def get_activation(activation_name):
    try:
        activation = activations[activation_name]
    except ValueError:
        activation = eval(activation)
    return activation


class FullyConnectedBlock(torch.nn.Module):
    
    def __init__(
        units, 
        activations, 
        input_shape=None, 
        output_shape=None, 
        dropouts=None, 
        name=None
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        assert len(units) == len(activations)
        if dropouts:
            assert len(dropouts) == len(units)
    
        activations = [get_activation(a) for a in activations]
        self.layers = []
    
        for i in range(len(units)):
            if i == 0 and input_shape:
                self.layers.append(torch.nn.Linear(in_features=input_shape, out_features=units[i]))
            self.layers.append(torch.nn.Linear(in_features=units[i-1], out_features=units[i]))
            
            self.layers.append(activations[i])
    
            if dropouts and dropouts[i]:
                self.layers.append(torch.nn.Dropout(p=dropouts[i]))
         
                
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor
        for layer in self.layers:
            x = layer(x)
        
        if self.output_shape:
            torch.reshape(x, shape=self.output_shape)
        return x


class SingleBlock(torch.nn.Module):
    
    def __init__(
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
            
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.linear(input_tensor)
        if self.batchnorm:
            x = self.batchnorm_layer(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout_layer(x)
        return x


class FullyConnectedBlock(torch.nn.Module):
    
    def __init__(
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
        self.blocks = []
        
        for i, (act, dropout) in enumerate(zip(activations, dropouts)):
            self.blocks.append(SingleBlock(
                input_shape=(units if i > 0 else input_shape), 
                output_shape=units, 
                activations=act, 
                batchnorm=batchnorm, 
                dropout=dropout,
            ))


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor: 
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
        input1_shape, 
        input2_shape, 
        reshape_input1=None, 
        reshape_input2=None, 
        dim=-1, 
        name=None
    ) -> None:
        super().__init__()
        selp.input1_shape = input1_shape
        selp.input1_shape = input1_shape
        self.reshape_input1 = reshape_input1
        self.reshape_input2 = reshape_input2
        self.dim = dim
        
        
    def forward(self, in1: torch.Tensor, in2: torch.Tensor) -> torch.Tensor:
        concat1, concat2 = in1, in2
        if self.reshape_input1:
            concat1 = torch.reshape(concat1, self.reshape_input1)
        if self.reshape_input2:
            concat2 = torch.reshape(concat2, self.reshape_input2)
        out = torch.cat((concat1, concat2), dim=self.dim)
        return out


class ConvBlock(torch.nn.Module):

    def __init__(
        filters,
        kernel_sizes,
        paddings,
        activations,
        poolings,
        kernel_init='glorot_uniform',
        input_shape=None,
        output_shape=None,
        dropouts=None,
        name=None,
    ) -> None:
        super().__init__()
        assert len(filters) == len(kernel_sizes) == len(paddings) == len(activations) == len(poolings)
        if dropouts:
            assert len(dropouts) == len(filters)
    
        activations = [get_activation(a) for a in activations]
        self.layers = []
        
        for i, (nfilt, ksize, padding, act, pool) in enumerate(zip(filters, kernel_sizes, paddings, activations, poolings)):
            self.layers.append(torch.nn.Conv2D(
                in_channels=(input_shape if (i == 0 and input_shape) else nfilters[i-1]),
                out_channels=nfilt,
                kernel_size=ksize,
                padding=padding,
            ))
            
            self.layers.append(act)
    
            if dropouts and dropouts[i]:
                self.layers.append(torch.nn.Dropout(p=dropouts[i]))
    
            if pool:
                self.layers.append(torch.nn.MaxPool2D(kernel_size=pool))
                
                
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor: 
        x = input_tensor
        for layer in self.layers:
            x = layer(x)
            
        if output_shape:
            torch.reshape(x, output_shape)
        return x


class VectorImgConnectBlock(torch.nn.Module):
    
    def __init__(
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
    
        assert len(vector_shape) == 1
        assert 2 <= len(img_shape) <= 3
    
    
    def forward(self, input_vec: torch.Tensor, input_img: torch.Tensor) -> torch.Tensor:
        block_input = input_img
        if len(self.img_shape) == 2:
            block_input = torch.reshape(block_input, img_shape + (1,))
        if not vector_bypass:
            reshaped_vec = torch.tile(torch.reshape(input_vec, (1, 1) + vector_shape), dims=(1, *img_shape[:2], 1))
            block_input = torch.cat((block_input, reshaped_vec), dim=-1)
    
        block_output = self.block(block_input)
    
        outputs = [input_vec, block_output]
        if concat_outputs:
            outputs = torch.cat(outputs, dim=-1)
        return outputs


def build_block(block_type, arguments):
    if block_type == 'fully_connected':
        block = fully_connected_block(**arguments)
    elif block_type == 'conv':
        block = conv_block(**arguments)
    elif block_type == 'connect':
        inner_block = build_block(**arguments['block'])
        arguments['block'] = inner_block
        block = vector_img_connect_block(**arguments)
    elif block_type == 'concat':
        block = concat_block(**arguments)
    elif block_type == 'fully_connected_residual':
        block = fully_connected_residual_block(**arguments)
    else:
        raise (NotImplementedError(block_type))

    return block


class FullModel(nn.Module):
    
    def __init__(
        block_descriptions,
        name=None, 
        custom_objects_code=None,
    ) -> None:
        super().__init__()
        if custom_objects_code:
            print("build_architecture(): got custom objects code, executing:")
            print(custom_objects_code)
            exec(custom_objects_code, globals(), custom_objects)
    
        self.blocks = [build_block(**descr) for descr in block_descriptions]
    
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        outputs = inputs
        for block in blocks:
            outputs = block(outputs)
        return outputs
