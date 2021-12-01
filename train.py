import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NoisyLibriSpeechDataset
from display import play_audio, plot_loss, display_results
from models import FCAE, CDAE, UNet


def train(
    device, model, dataloader,
    epochs=100, learning_rate=1e-3, criterion=nn.MSELoss()):

    # Initialize training history
    history = {
        'losses': [],
        'times': [],
        'params': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'critetion': critetion._get_name
        }
    }

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5)

    # Initalize progress bar
    pbar = tqdm(total=epochs, ascii=True, ncols=119)

    # Loop over epochs
    for epoch in range(epochs):
        history['times'].append(time.time())

        # Set the model in training mode
        model.train()
        total_train_loss = 0

        # Loop over batches
        for batch in dataloader:

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


        history['losses'].append(total_train_loss)
        history['times'][-1] = time.time() - history['times'][-1]

        pbar.set_postfix({
            'avg. loss':  np.mean(history['losses']),
            'cur. loss':  history['losses'][-1]})
        pbar.update(1)

    return model, history


def evaluate(model, data_test):
    sample = data_test[0]
    sample['denoised_magnitude'] = data_test.restore(model.forward(
        sample['magnitude']).squeeze())
    sample = data_test.restore(sample)

    clean_waveform = data_test.libri[sample['libri_index']][0].numpy()[0]
    noisy_waveform = data_test.spec_to_wav(
        sample['magnitude'], sample['phase'])
    denoised_waveform = data_test.spec_to_wav(
        sample['denoised_magnitude'], sample['phase'])

    fig, axes = display_results(
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
    seed = 1
    batch_size=8

    N = 2703
    N = 5
    test_size = .10
    conv = False

    # Create dataset splits
    train_idxs, val_idxs, test_idxs = get_data_split_idxs(
        N, test_size=.10, seed=seed)

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
        device,
        model,
        train_dl,
        epochs=epochs,
        learning_rate=learning_rate,
        criterion=loss)

    # Plot Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plot_loss(ax, hist['losses'], 'MSE')
    fig.show()

    # Evaluate Model
    fig, axes = evaluate(model, data_test)
    fig.show()
