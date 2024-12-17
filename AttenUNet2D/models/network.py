# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:00:42 2024

@author: user
"""
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead

# Build model
class UNet2D(nn.Module):
    def __init__(self, encoder_name='resnet50', encoder_depth=5, encoder_weights='imagenet', 
                 in_channels=3, decoder_channels = (256, 128, 64, 32, 16), attention_type = 'se', classes=3):
        super(UNet2D, self).__init__()
        
        '''Encoder'''
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        
        '''Decoder'''
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=attention_type
            )
        
        '''Segmentation head'''
        # segmentation head
        self.segmentation = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=3
            )
        
    def forward(self, x):
        # Encoder 
        encoder_features = self.encoder(x)
        
        # Decoder
        decoder_features = self.decoder(*encoder_features)
        seg = self.segmentation(decoder_features)
        
        return seg
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)