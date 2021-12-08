import numpy as np
from torch import nn


class CDAE(nn.Module):
    def __init__(self, n_layers=3, z_dim=8, in_channels=1, batch_norm=False):
        super(CDAE, self).__init__()

        self.name = 'Convolutional Autoencoder'

        # Define activation "layer"
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        z_coef = self._nearest_2pow_coef(z_dim) + 1
        conv_sizes = [2**i for i in range(z_coef, z_coef + n_layers - 1)]
        conv_sizes = [z_dim] + conv_sizes + [in_channels]
        encoder_layers = []
        for i in range(len(conv_sizes) - 1):
            # Conv layer
            encoder_layers.append(nn.Conv2d(
                in_channels=conv_sizes[-1-i],
                out_channels=conv_sizes[-1-i-1],
                kernel_size=3,
                padding='same'))
            # Relu activation
            encoder_layers.append(self.relu)
            # Max pooling layer
            encoder_layers.append(self.pool)
            # Batch normalization layer
            if (batch_norm):
                encoder_layers.append(nn.BatchNorm2d(conv_sizes[-1-i-1]))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(conv_sizes) - 1):
            decoder_layers.append(nn.ConvTranspose2d(
                in_channels=conv_sizes[i],
                out_channels=conv_sizes[i+1],
                kernel_size=2,
                stride=2))
            decoder_layers.append(self.relu)

        self.decoder = nn.Sequential(*decoder_layers)


    def __repr__(self):
        return self.name + '()'

    def __str__(self):
        return self.name + '()'


    def _get_layer_dims(self, n_layers, z_dim, in_channels):
        z_coef = self._nearest_2pow_coef(z_dim) + 1
        conv_sizes = [2**i for i in range(z_coef, z_coef + n_layers - 1)]
        conv_sizes = [z_dim] + conv_sizes + [in_channels]
        return conv_sizes


    def _nearest_2pow(self, x):
        return 2**np.ceil(np.log10(x)/np.log10(2)).astype('int')


    def _nearest_2pow_coef(self, x):
        return np.log2(self._nearest_2pow(x)).astype('int')


    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        y = self.sigmoid(y)
        return y
