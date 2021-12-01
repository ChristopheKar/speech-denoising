import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import display
from dataset import NoisyLibriSpeechDataset, utils
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


def evaluate(device, model, data_test):
    sample = data_test[0]
    sample['denoised_magnitude'] = data_test.restore(model.forward(
        sample['magnitude'].to(device)).squeeze())
    sample = data_test.restore(sample)

    clean_waveform = data_test.libri[sample['libri_index']][0].numpy()[0]
    noisy_waveform = data_test.spec_to_wav(
        sample['magnitude'], sample['phase'])
    denoised_waveform = data_test.spec_to_wav(
        sample['denoised_magnitude'], sample['phase'])

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

    # Data params
    srate = 16000
    data_root = 'data/noised_synth_babble'
    libri_root = 'data/LibriSpeech/dev-clean'
    seed = 11
    batch_size=8

    N = 20
    test_size = .10
    conv = False

    # Create dataset splits
    train_idxs, val_idxs, test_idxs = utils.get_data_split_idxs(
        N, test_size=test_size, seed=seed)

    # Load training data
    data_train = NoisyLibriSpeechDataset(
        data_root=data_root, libri_root=libri_root,
        include_idxs=train_idxs, test=False,
        conv=conv, seed=seed)
    train_dl = DataLoader(
        data_train, batch_size=batch_size,
        num_workers=0, pin_memory=False)

    # Load validation data
    data_val = NoisyLibriSpeechDataset(
        data_root=data_root, libri_root=libri_root,
        include_idxs=val_idxs, test=False,
        conv=conv, seed=seed)
    val_dl = DataLoader(
        data_val, batch_size=batch_size,
        num_workers=0, pin_memory=False)

    # Load testing data
    data_test = NoisyLibriSpeechDataset(
        data_root=data_root, libri_root=libri_root,
        include_idxs=test_idxs, test=True,
        conv=conv, seed=seed)

    display.show_split_sizes((data_train, data_val, data_test))

    # Model params
    loss = nn.MSELoss()
    epochs = 2
    learning_rate = 0.01

    # Create model and send to device
    model = FCAE(
        data_train.target_shape,
        n_layers=4,
        z_dim=8).to(device)

    # Train model
    model, hist = train(
        device, model,
        train_dl, val_dl,
        epochs=epochs,
        learning_rate=learning_rate,
        criterion=loss)

    # Plot Losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = display.plot_losses(ax, hist, 'MSE')
    fig.show()

    # Evaluate Model
    fig, axes = evaluate(device, model, data_test)
    fig.show()
