import numpy as np
from torch import nn


class FCAE(nn.Module):
    """Implements a basic fully-connected autoencoder network"""

    def __init__(self, in_shape=(256, 256), n_layers=3, z_dim=64, dropout=0.):
        super(FCAE, self).__init__()

        # Define input dimensions
        self.in_shape = in_shape
        self.n_input = in_shape[0]*in_shape[1]
        self.flatten = nn.Flatten()

        # Get layer dimensions
        layer_sizes = self._get_layer_dims(n_layers, z_dim, self.n_input)

        # Define ReLU "layer"
        self.relu = nn.ReLU()

        # Encoder
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(
                layer_sizes[-1-i],
                layer_sizes[-1-i-1]))
            encoder_layers.append(self.relu)

        # Insert dropout layer before final encoder layer
        if (dropout > 0):
            encoder_layers.insert(-1, nn.Dropout(dropout))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(layer_sizes) - 1):
            decoder_layers.append(nn.Linear(
                layer_sizes[i],
                layer_sizes[i+1]))
            decoder_layers.append(self.relu)

        self.decoder = nn.Sequential(*decoder_layers)


    def _get_layer_dims(self, n_layers, z_dim, n_input):
        z_coef = self._nearest_2pow_coef(z_dim) + 1
        conv_sizes = [2**i for i in range(z_coef, z_coef + n_layers - 1)]
        conv_sizes = [z_dim] + conv_sizes + [n_input]
        return conv_sizes


    def _nearest_2pow(self, x):
        return 2**np.ceil(np.log10(x)/np.log10(2)).astype('int')


    def _nearest_2pow_coef(self, x):
        return np.log2(self._nearest_2pow(x)).astype('int')


    def forward(self, x):

        # Flatten input
        x = self.flatten(x)
        # Encoder Chain
        z = self.encoder(x)
        # Decoder Chain
        x = self.decoder(z)
        # Reshape output
        out = x.reshape((-1, *self.in_shape))
        return out
