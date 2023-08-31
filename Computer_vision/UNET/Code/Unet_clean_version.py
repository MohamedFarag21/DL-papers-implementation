# Importing the necessary libraries for the project

# torch is the main PyTorch library
import torch

# torchvision.transforms provides common image transformations
import torchvision.transforms as transforms

# PIL.Image module provides a class with the same name which is used to represent a PIL image
from PIL import Image

# torchvision.models.segmentation provides pre-trained segmentation models
from torchvision.models import segmentation

# torch.utils.data provides utilities for loading and manipulating data
from torch.utils.data import Dataset, DataLoader, random_split

# pandas is a data manipulation and analysis library
import pandas as pd

# os is a module providing functions for interacting with the operating system
import os

# tifffile is a library to work with TIFF files
import tifffile as tiff

# cv2 (OpenCV) is a library of programming functions mainly aimed at real-time computer vision
import cv2

# tqdm is a library for creating progress bars
from tqdm import tqdm

# numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices
import numpy as np

# torch.nn provides classes to build neural networks
import torch.nn as nn

# torch.nn.functional provides functions that don't have any parameters
import torch.nn.functional as F




def convblock(in_channels, out_channels, kernel_size = (3,3), padding='same'):
    """
    This function defines a convolutional block used in the U-Net architecture.
    It consists of two convolutional layers, each followed by a batch normalization and a ReLU activation function.
    
    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    :param kernel_size: The size of the kernel used in the convolution operation, default is (3,3).
    :param padding: The type of padding operation to be used, default is 'same'.
    
    :return conv: A Sequential module that includes the structure of the block.
    """
    
    # Define the convolutional block
    conv = nn.Sequential(
        # First convolutional layer
        nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, padding=padding),
        # Batch normalization after the first convolution
        nn.BatchNorm2d(num_features=out_channels),
        # ReLU activation function after the first batch normalization
        nn.ReLU(),
        # Second convolutional layer
        nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= kernel_size, padding=padding),
        # Batch normalization after the second convolution
        nn.BatchNorm2d(num_features=out_channels),
        # ReLU activation function after the second batch normalization
        nn.ReLU(),
    )
    
    # Return the defined convolutional block
    return conv
    



def deconvblock(in_channels, out_channels):
    """
    This function defines a deconvolutional block used in the U-Net architecture.
    It consists of a convolutional layer followed by a batch normalization and a ReLU activation function.
    
    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    
    :return deconv: A Sequential module that includes the structure of the block.
    """
    
    # Define the deconvolutional block
    deconv = nn.Sequential(
        # Convolutional layer with a kernel size of 2x2 and 'same' padding
        nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= (2,2), padding='same'),
        # Batch normalization after the convolution
        nn.BatchNorm2d(num_features=out_channels),
        # ReLU activation function after the batch normalization
        nn.ReLU()
    )
    
    # Return the defined deconvolutional block
    return deconv



class unet(nn.Module):
    
    def __init__(self, input_channels, output_classes, channel_list):
        # call the parent constructor
        super(unet, self).__init__()

        # Encoder Layers
        self.channel_list = channel_list
        self.input_channels = input_channels

        self.conv_blocks =  nn.Sequential(*[convblock(self.channel_list[i], self.channel_list[i+1]) 
                                            if i != len(self.channel_list) -1 
                                            else None  for i in range(len(self.channel_list))])
        

        # Space holders for convlution feature maps & pooled outputs, if you want to do any extra computation
        self.feature_map = {}
        self.pooled_outputs={}

        # Maxpooling layer
        self.maxpool2D = nn.MaxPool2d((2,2))

        # Upsampling layer
        self.upsampling2D = nn.Upsample(scale_factor=2, mode='bilinear')

        
        # Decoder layers

        # Deconvblocks layers
        self.deconv_blocks = nn.Sequential(*[deconvblock(self.channel_list[i], self.channel_list[i-1]) 
                                            if i !=  (len(self.channel_list) - len(self.channel_list))+1 
                                            else  None for i in reversed(range(len(self.channel_list)))])
        # # Conv layers in decoder
        self.conv_decoder = nn.Sequential(*[convblock(self.channel_list[i], self.channel_list[i-1]) 
                                            if i !=  (len(self.channel_list) - len(self.channel_list))+1 
                                            else  None for i in reversed(range(len(self.channel_list)))])
        

        # Output layer
        self.output = nn.Conv2d(in_channels=64, out_channels=output_classes, kernel_size=(1,1))


        

    def forward(self, x):
        """
        Performs the forward pass of the U-Net model.

        :param x: The input tensor.

        :return: The output tensor.
        """

        # Encoder part

        # Initialize the pooling operation with the input tensor
        pool = x
        
        # print(self.conv_blocks[i])

        # Loop over each channel in the channel list
        for i in range(len(self.channel_list)):
            # We want to perform the sequence: input -> convolution -> pooling for each layer,
            # but for the last layer we only need the convolution operation without pooling
            # print(self.conv_blocks[i])
            # break
            if (self.conv_blocks[i] != None):
                # print(i)
                # Apply the i-th convolution block to the pooled output and store the result in the feature_maps dictionary
                # The key is 'conv' concatenated with the string representation of the channel number
                
                # print(self.conv_blocks)
                self.feature_map['conv' + str(self.channel_list[i+1])] = self.conv_blocks[i](pool)
                
                # Apply the maxpool2D operation to the convolution result of the current channel
                # The result is stored in the pooled_outputs dictionary
                # The key is 'pool' concatenated with the string representation of the channel number
                self.pooled_outputs['pool' + str(self.channel_list[i+1])] = self.maxpool2D(self.feature_map['conv' + str(self.channel_list[i+1])])
                
                # Update the pooled output to be used as the input for the next iteration
                pool = self.pooled_outputs['pool' + str(self.channel_list[i+1])]

                # print(self.feature_map.keys())
                # print(self.pooled_outputs.keys())
            else:
                None 

        bottleneck = self.feature_map['conv' + str(self.channel_list[-1])]

        # Decoding step

        # Loop over each channel in the channel list
        for i in range(len(self.channel_list)):
            # Check if both the convolution block and deconvolution block are not None
            if (self.conv_blocks[i] != None) and (self.deconv_blocks[i] != None):
                # print(self.deconv_blocks)
                
                # Upsample the bottleneck feature map
                upsample = self.upsampling2D(bottleneck)
                
                # Apply the deconvolution block to the upsampled feature map
                deconv = self.deconv_blocks[i](upsample)
                
                # Concatenate the deconvolved feature map with the corresponding convolution feature map
                concat = torch.cat([deconv, self.feature_map['conv'+ str(self.channel_list[-2-i])]], dim=1)
                # print(concat.shape)
                
                # Apply the convolution block to the concatenated feature map
                bottleneck = self.conv_decoder[i](concat)
                # print(concat.shape)
            else:
                None   
            



        output = self.output(bottleneck)
        # print('output: ', output.shape)
        # Dimensions: (512, 512, 64) ---> output ---> (512, 512, 43)
        
        return output
    


#%%
# The advantage of this implementation that you can change the input channels, the output channels, also the architecture of the network
test = torch.rand(32, 10, 512, 512)
input_channels = 10
output_classes = 3
channel_list = [input_channels, 64, 128, 256, 512]

model = unet(input_channels=input_channels, output_classes=output_classes, channel_list=channel_list)

result = model(test)

# import wandb

# # Initialize Weights and Biases
# wandb.init()

# # Load your model
# # wandb login

# # Measure the inference time
# with torch.no_grad():
#     # Start the timer
#     start_time = torch.cuda.Event(enable_timing=True)
#     end_time = torch.cuda.Event(enable_timing=True)
#     start_time.record()

#     # Run the inference
#     output = model(test)

#     # End the timer
#     end_time.record()
#     torch.cuda.synchronize()

#     # Calculate the inference time in milliseconds
#     inference_time = start_time.elapsed_time(end_time)

#     # Log the inference time to Weights and Biases
#     wandb.log({"Inference Time (ms)": inference_time})

# Finish the Weight


# print(model.conv_blocks[0])

#%%
# print(model.architecture)
# %%
# no_channels = [3 ,64, 128, 256, 512]
# conv_blocks = [convblock(no_channels[i], no_channels[i+1]) if i != len(no_channels) -1 else  None for i in range(len(no_channels))]
# print(conv_blocks)