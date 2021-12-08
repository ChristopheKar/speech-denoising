import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

import torch
import torch.nn as nn

import display
from dataset.libri import load_data
from models import FCAE, CDAE, UNet


def train(
    device, model, train_dl, val_dl,
    epochs=100, learning_rate=1e-3, criterion=nn.MSELoss()):

    # Initialize training history
    history = {
        'losses': [],
        'val_losses': [],
        'times': [],
        'params': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'critetion': criterion._get_name
        }
    }

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5)

    # Check if environment is notebook
    try:
        get_ipython
        is_notebook = True
    except:
        is_notebook = False

    # Initalize progress bars depending on environment
    if (is_notebook):
        epoch_pbar = tqdm_nb(total=epochs, desc='Epochs')
        train_pbar = tqdm_nb(total=len(train_dl), desc='Training Batches')
        val_pbar = tqdm_nb(total=len(val_dl), desc='Validation Batches')
    else:
        epoch_pbar = tqdm_nb(
            total=epochs, desc='Epochs', ascii=True, ncols=159)
        train_pbar = tqdm_nb(
            total=len(train_dl),
            desc='Training Batches',
            ascii=True, ncols=159)
        val_pbar = tqdm_nb(
            total=len(val_dl),
            desc='Validation Batches',
            ascii=True, ncols=159)

    # Loop over epochs
    for epoch in range(epochs):
        history['times'].append(time.time())

        # Set the model in training mode
        model.train()
        total_train_loss = 0

        # Training Loop
        train_pbar.reset()
        for batch in train_dl:

            # Send inputs to device
            x = batch['magnitude'].to(device)
            y = batch['target'].to(device)

            # Forward pass and loss
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # Compute gradient
            loss.backward()

            # Update model parameters and
            # zero out previously accumulated gradients
            optimizer.step()
            optimizer.zero_grad()

            # Update training loss
            total_train_loss += loss.item()

            # Update progress bar
            if (epoch == 0):
                mean_train_loss = total_train_loss
            else:
                mean_train_loss = np.mean(history['losses'])
            train_pbar.set_postfix({
                'avg. train loss':  mean_train_loss,
                'cur. total loss': total_train_loss,
                'cur. batch loss':  loss.item()
            })
            train_pbar.update(1)

        # Empty GPU cache
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        # Set model for evaluation
        model.eval()
        total_val_loss = 0

        # Validation Loop
        val_pbar.reset()
        for batch in val_dl:
            # Send inputs to device
            x = batch['magnitude'].to(device)
            y = batch['target'].to(device)

            # Forward pass and loss
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # Update validation loss
            total_val_loss += loss.item()

            # Update progress bar
            if (epoch == 0):
                mean_val_loss = total_val_loss
            else:
                mean_val_loss = np.mean(history['val_losses'])
            val_pbar.set_postfix({
                'avg. valid loss': mean_val_loss,
                'cur. total loss': total_val_loss,
                'cur. batch loss': loss.item()
            })
            val_pbar.update(1)

        # Update history
        history['losses'].append(total_train_loss)
        history['val_losses'].append(total_val_loss)
        history['times'].append(time.time())
        # Update progress bar
        epoch_pbar.update(1)


    return model, history


def evaluate(device, model, data):
    # Load test sample
    sample = data[0]
    # Predict denoised magnitude
    sample['denoised_magnitude'] = data.restore(model.forward(
        sample['magnitude'].to(device)).squeeze())
    # Inverse transform sample
    sample = data.restore(sample)

    # Load clean waveform signal and pad
    clean_waveform = data.libri[sample['libri_index']][0].numpy()[0]
    pad_len = sample['magnitude'].shape[0]*data.srate
    pad_len = pad_len - clean_waveform.shape[0]
    clean_waveform = np.pad(clean_waveform, (0, pad_len))

    # Reconstruct noised and denoised waveforms
    noisy_waveform = data.spec_to_wav(
        sample['magnitude'], sample['phase'])
    denoised_waveform = data.spec_to_wav(
        sample['denoised_magnitude'], sample['phase'])

    # Show results
    fig, axes = display.show_results(
        clean_waveform, noisy_waveform, denoised_waveform)

    return fig, axes


def set_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if (verbose):
        print('device: ', device)

    return device


if __name__ == '__main__':

    # Set compute device
    device = set_device(verbose=True)

    params = {
        'network': UNet,
        'data': {
            'N': 10,
            'test_size': .10,
            'data_root': 'data/noised_synth_babble',
            'libri_root': 'data/LibriSpeech/dev-clean',
            'batch_size': 8,
            'pin_memory': (device == 'cuda'),
            'conv': True,
            'seed': 1,
            'srate': 16000
        },
        'model': {
            'in_shape': (256, 256),
            'in_channels': 1,
            'n_classes': 1,
            'encoder_channels': (4, 8, 16),
            'decoder_channels': (16, 8, 4),
            'retain_dim': True
        },
        'train': {
            'epochs': 2,
            'learning_rate': 0.001,
            'criterion': nn.BCEWithLogitsLoss()
        }
    }

    model = params['network'](**params['model']).to(device)
    print(model)
    print('---')

    print('\nLoading data...\n')
    data_train, train_dl, data_val, val_dl, data_test = load_data(**params['data'])
    display.show_split_sizes((data_train, data_val, data_test))

    print('\nTraining model...\n')
    model, hist = train(device, model, train_dl, val_dl, **params['train'])

    # Plot Losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = display.plot_losses(ax, hist, repr(params['train']['criterion']))
    fig.show()

    # Evaluate Model
    fig, axes = evaluate(device, model, data_test)
    fig.show()
