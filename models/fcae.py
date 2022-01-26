import torch
from torch import nn


class FCAE(nn.Module):
    """Implements a basic fully-connected autoencoder network"""

    def __init__(
        self, in_shape=(256, 1),
        encoder_dims=[128, 64, 32], decoder_dims=[32, 64, 128],
        z_dim=16, out_activation='relu', dropout=0.):
        super(FCAE, self).__init__()

        self.name = 'Fully Connected Autoencoder'

        # Define input dimensions
        self.in_shape = in_shape
        in_shape = in_shape + [1]
        self.n_input = in_shape[0]*in_shape[1]
        self.flatten = nn.Flatten()

        # Get layer dimensions
        encoder_dims = [self.n_input] + encoder_dims + [z_dim]
        decoder_dims = [z_dim] + decoder_dims + [self.n_input]

        # Define activation "layers"
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if (out_activation == 'relu'):
            self.out_activation = self.relu
        else:
            self.out_activation = self.sigmoid

        # Encoder
        encoder_layers = []
        for i in range(1, len(encoder_dims)):
            encoder_layers.append(nn.Linear(
                encoder_dims[i-1],
                encoder_dims[i]))
            encoder_layers.append(self.relu)

        # Insert dropout layer before final encoder layer
        if (dropout > 0):
            encoder_layers.insert(-1, nn.Dropout(dropout))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(1, len(decoder_dims)):
            decoder_layers.append(nn.Linear(
                decoder_dims[i-1],
                decoder_dims[i]))
            decoder_layers.append(self.relu)

        decoder_layers[-1] = self.out_activation
        self.decoder = nn.Sequential(*decoder_layers)


    def __repr__(self):
        return self.name


    def __str__(self):
        return self.name


    def forward(self, x):

        preds = torch.zeros_like(x)
        for t in range(x.shape[-1]):
            preds[:, :, t] = self.decoder(self.encoder(x[:, :, t]))

        return preds
