import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        """Block forward pass."""
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):

    def __init__(self, channels=(1, 16, 32, 64)):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Block forward pass."""
        block_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)

        return block_outputs


class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                 for i in range(len(channels) - 1)])
        self.decoder_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])

    def forward(self, x, encoder_features):
        """Block forward pass."""
        for i in range(len(self.channels) - 1):
            # Upsampling convolutions
            x = self.upconvs[i](x)
            # Manually center-crop the features
            encoder_feature = self.crop(encoder_features[i], x)
            # Concatenate encoder features with upsampled features
            x = torch.cat([x, encoder_feature], dim=1)
            # Pass through decoder block
            x = self.decoder_blocks[i](x)

        return x

    def crop(self, features, x):
        """Center crop `features` to match `x` shape."""
        h = (features.shape[-2] - x.shape[-2])//2
        w = (features.shape[-1] - x.shape[-1])//2
        return features[:, :, h:-h, w:-w]


class UNet(nn.Module):
    def __init__(
        self,
        in_shape=(256, 256),
        in_channels=1,
        n_classes=1,
        encoder_channels=(16, 32, 64),
        decoder_channels=(64, 32, 16),
        retain_dim=True):

        super().__init__()

        self.name = 'U-Net'

        # initialize the encoder and decoder
        encoder_channels = [in_channels] + list(encoder_channels)
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(decoder_channels[-1], n_classes, 1)
        self.retain_dim = retain_dim
        self.input_shape = in_shape


    def __repr__(self):
        return self.name + '()'


    def __str__(self):
        return self.name + '()'


    def forward(self, x):
        """Block forward pass."""
        encoder_features = self.encoder(x)
        decoder_features = self.decoder(
            encoder_features[::-1][0], encoder_features[::-1][1:])
        # Pass through regression head for segmentation
        segmentation_map = self.head(decoder_features)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if (self.retain_dim):
            segmentation_map = F.interpolate(
                segmentation_map, self.input_shape)

        return segmentation_map
