# Speech Denoising using Deep Learning

This project aims at exploring some basic deep learning techniques to denoise speech,
using the [PyTorch](https://pytorch.org/) framework.

## Installation

### Dependencies

This project uses Python 3, and the dependencies listed in `requirements.txt` can
be installed with `pip`, `conda`, or any other package manager, in a virtual environment
or other. For example, using `pip`:
```bash
pip install -r requirements.txt
```

**Note:** This repo also uses Jupyter notebooks but that dependency is not
listed in requirements.txt.

## Venv

You can create a virtual environment using `venv` and install the dependencies, for example:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Docker

The environment can also be setup using Docker and the provided `Dockerfile`.
First, build the image by running the following command in this repository:
```bash
docker build -t speech_denoising .
```

Then, using the built image is as simple as:
```bash
docker run -it --rm --name speech -v $PWD:/workspace speech_denoising bash
```

This will drop you in an interactive bash session inside the Docker container.\
The `-v` option allows you to mount your current workspace to `/workspace`
inside the container, so that your files are accessible from there, and so that any
changes made to files under that path persists on your local storage. Any other changes
made inside the Docker container, e.g. installing additional packages or creating files
outside of `/workspace`, will not persist across sessions unless you commit your changes
to the Docker image.


## Usage

### Dataset

The dataset used in this project is the [LibriSpeech](https://www.openslr.org/12/) dataset.

The code is tested on the `dev-clean` subset, but any subset should work. Download
the LibriSpeech subset into the data directorty as such:

```bash
# Change to data directory
cd data
# Download subset
curl -O https://www.openslr.org/resources/12/dev-clean.tar.gz
# Extract files from archive
tar -zxvf dev-clean.tar.gz
```

### Training

The training script takes in a configuration file in JSON format containing data, model, and training parameters. Example configurations for the three types of available models are available in configs/. To run a basic fully-connected autoencoder, run:
```bash
python train.py --config configs/fcae.json
```

## Code Structure

This repository's structure is fairly straightforward.

- `data/` contains the needed datasets, which can be downloaded as per the instructions above, nothing else to see here.
- `models/` is a module containing different network architectures, namely autoencoders with different architectures (fully-connected, convolutional, UNet). Each model is implemented as a class in its own file.
- `configs/` contains JSON configuraiton files used to speed up the experimentation and iteration process. All you need to do is change parameters in the configuration files to change any major aspect in the training, such as network to use, loss function, number of epochs, amount of data, etc.
- `utils/` contains modules necessary for all aspects of the pipeline besides the training. For example, `data.py` contains functions and classes to handle data creation and custom data loaders for PyTorch. Other scripts are used to define the metrics used for evaluation, utility display functions, and most importantly, `transforms.py` contains classes defining all the transforms used in the data pipeline: scaling, log-transforms, and STFTs.
- Finally, `train.py` contains the main functions for the training and evaluation loop, and can be run directly as a script that takes a config path as a parameter. It will then load the settings from the configuration file, run the training, and save the results in the output directory, usually `results/`.

Everything should be quite modular, the datasets can be easily change, for example for different LibriSpeech subsets, and everything is separated enough so that replacing a dataset only leads to rewriting a small data loading function instead of entire data loaders.
