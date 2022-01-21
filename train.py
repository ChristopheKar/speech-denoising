import os
import json
import time
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
import soundfile as sf
import torch
import torch.nn as nn

import models
from models import FCAE, CDAE, UNet
from utils import display
from utils import transforms
from utils import metrics
from utils.data import load_data
from utils.utils import set_seed, set_device


def load_params_from_config(config, device):
    """Load training parameters from JSON or dict config."""
    if (isinstance(config, str)):
        if (config.endswith('.json')):
            name = config.split('/')[-1].split('.json')[0]
            with open(config, 'r') as cfg_fo:
                config = json.load(cfg_fo)
        else:
            msg = (
                'If `config` path is provided,'
                'it must be a path to a JSON file.')
            raise TypeError(msg)
    elif (isinstance(config, dict)):
        name = config['network']
    else:
        msg = '`config` must be either dict or path to JSON file.'
        raise TypeError(msg)

    # Set config name
    config['name'] = config.get('name', name)

    # Set pin_memory according to device
    config['data']['pin_memory'] = (device == 'cuda')
    # Convert string inputs to module functions
    config['network'] = getattr(models, config['network'])
    config['train']['criterion'] = getattr(nn, config['train']['criterion'])()

    return config


def save_hist(hist, filepath):
    with open(filepath, 'wb') as pkl_fo:
        pickle.dump(hist, pkl_fo)


def load_hist(filepath):
    with open(filepath, 'rb') as pkl_fo:
        hist = pickle.load(pkl_fo)
    return hist


def train(
    device, model, name,
    train_dl, val_dl, output_dir='results',
    epochs=100, learning_rate=1e-3, criterion=nn.MSELoss()):

    # Initialize results output dir
    output_dir = os.path.join(output_dir, name)
    history_fp = os.path.join(output_dir, 'history.pkl')
    model_fp = os.path.join(output_dir, name + '.pth')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize training history
    history = {
        'dir': output_dir,
        'losses': [],
        'val_losses': [],
        'times': [],
        'params': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'criterion': criterion._get_name
        },
        'best_epoch': 0,
        'model_path': model_fp,
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
        epoch_pbar = tqdm(
            total=epochs, desc='Epochs', ascii=True, ncols=159)
        train_pbar = tqdm(
            total=len(train_dl),
            desc='Training Batches',
            ascii=True, ncols=159)
        val_pbar = tqdm(
            total=len(val_dl),
            desc='Validation Batches',
            ascii=True, ncols=159)

    # Loop over epochs
    try:
        for epoch in range(epochs):
            history['times'].append(time.time())

            # Set the model to training mode
            model.train()
            total_train_loss = 0

            # Training Loop
            train_pbar.reset()
            for batch in train_dl:

                # Send inputs to device
                x = batch['noised'].to(device)
                y = batch['clean'].to(device)

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
                x = batch['noised'].to(device)
                y = batch['clean'].to(device)

                # Forward pass and loss
                y_pred = model.forward(x)
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
            history['losses'].append(total_train_loss/len(train_dl))
            history['val_losses'].append(total_val_loss/len(val_dl))
            history['times'].append(time.time())

            # Save model and history
            if (epoch > 1):
                is_best = (history['val_losses'][-1] < history['val_losses'][-2])
            else:
                is_best = True
            if (is_best):
                history['best_epoch'] = epoch + 1
                torch.save(model.state_dict(), model_fp)
            save_hist(history, history_fp)

            # Update progress bar
            rec_losses = ['{:.3g}'.format(i) for i in history['val_losses'][-5:]]
            epoch_pbar.set_postfix({
                'best epoch': history['best_epoch'],
                'val. losses': ', '.join(rec_losses)
            })
            epoch_pbar.update(1)

    except KeyboardInterrupt:
        print('Ctrl-C Keyboard Interrupt Detected...')
        print('Stopping training and returning model and history')
        print()
        return model, history


    return model, history


def evaluate(
    device, model,
    data, loader,
    fig=True, audio=True,
    export_prefix=None, verbose=True):

    scores = {'rmse': [], 'gain': []}

    for batch in loader:
        x = batch['noised'].to(device)
        y = batch['clean'].to(device)
        y_pred = model(x)

        for i in range(y_pred.shape[0]):
            mag = data.inv_transform(y_pred[i])
            phase = data.tensor.revert(batch['phase'][i])
            cleaned_wav = transforms.mag_phase_to_wav(mag, length=data.srate)
            clean_wav = data.tensor.revert(batch['waveforms']['clean'][i])
            noise_wav = data.tensor.revert(batch['waveforms']['noise'][i])
            scores['gain'].append(metrics.gain(
                clean_wav, noise_wav, cleaned_wav))
            scores['rmse'].append(metrics.rmse(
                clean_wav, cleaned_wav))

    scores['rmse'] = np.mean(scores['rmse'])
    scores['gain'] = np.mean(scores['gain'])

    if (verbose):
        for key, score in scores.items():
            print('{}: {:.3f}'.format(key, score))

    if (fig):
        fig, axes = plt.subplots(1, 3, figsize=(9, 5))
        axes[0].imshow(data.tensor.revert(x[i]), interpolation='none')
        axes[0].set_title('Clean Magnitude')
        axes[1].imshow(data.tensor.revert(y[i]), interpolation='none')
        axes[1].set_title('Noised Magnitude')
        axes[2].imshow(mag, interpolation='none')
        axes[2].set_title('Denoised Magnitude')
        if (export_prefix):
            fig.savefig(export_prefix + 'magnitudes.jpg')
        fig.show()

    if (audio):
        wavs = [clean_wav, batch['waveforms']['noised'][i], cleaned_wav]
        names = ['Original', 'Noised', 'Denoised']
        for name, wav in zip(names, wavs):
            print(name + ' Signal')
            display.play_audio(wav)
            if (export_prefix):
                display.save_audio(
                    export_prefix + name.lower() + '.wav', wav, data.srate)

    return scores


if __name__ == '__main__':

    # Argument parser for config path
    parser = argparse.ArgumentParser(
        description='Train speech denoiser.')
    parser.add_argument(
        '-c', '--config', required=True,
        help='Path to JSON config file for training. Examples in configs/')
    args = parser.parse_args()

    # Set compute device
    device = set_device(verbose=True)

    params = load_params_from_config(args.config, device)

    model = params['network'](**params['model']).to(device)
    print(model)
    print('---')

    print('\nLoading data...\n')
    train_data, train_dl, val_data, val_dl, test_data, test_dl = load_data(
        **params['data'])

    print('\nTraining model...\n')
    model, hist = train(
        device, model, params['name'],
        train_dl, val_dl,
        **params['train'])

    # Plot Losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = display.plot_losses(ax, hist, repr(params['train']['criterion']))
    fig.savefig(os.path.join(hist['dir'], 'losses.png'))
    fig.show()

    # Evaluate model
    scores = evaluate(
        device, model,
        train_data, train_dl,
        fig=True, audio=True,
        verbose=True,
        export_prefix=os.path.join(hist['dir'], 'train_'))

    # Evaluate model
    scores = evaluate(
        device, model,
        test_data, test_dl,
        fig=True, audio=True,
        verbose=True,
        export_prefix=os.path.join(hist['dir'], 'test_'))
