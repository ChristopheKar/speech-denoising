import numpy as np
from torch import nn


class CDAE(nn.Module):
    def __init__(
        self, in_channels=1,
        encoder_dims=[128, 64, 32], decoder_dims=[32, 64, 128],
        z_dim=16, out_activation='sigmoid', batch_norm=False):
        super(CDAE, self).__init__()

        self.name = 'Convolutional Autoencoder'

        # Define activation "layer"
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if (out_activation == 'relu'):
            self.out_activation = self.relu
        else:
            self.out_activation = self.sigmoid

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        encoder_dims = [in_channels] + encoder_dims
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            # Conv layer
            encoder_layers.append(nn.Conv2d(
                in_channels=encoder_dims[i],
                out_channels=encoder_dims[i+1],
                kernel_size=3,
                padding='same'))
            # Relu activation
            encoder_layers.append(self.relu)
            # Max pooling layer
            encoder_layers.append(self.pool)
            # Batch normalization layer
            if (batch_norm):
                encoder_layers.append(nn.BatchNorm2d(encoder_dims[i+1]))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_dims = decoder_dims + [in_channels]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.ConvTranspose2d(
                in_channels=decoder_dims[i],
                out_channels=decoder_dims[i+1],
                kernel_size=2,
                stride=2))
            decoder_layers.append(self.relu)

        decoder_layers[-1] = self.out_activation
        self.decoder = nn.Sequential(*decoder_layers)


    def __repr__(self):
        return self.name


    def __str__(self):
        return self.name


    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        y = self.sigmoid(y)
        return y
