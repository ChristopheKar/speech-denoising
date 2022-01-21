# Speech Denoising using Deep Learning

This project aims at exploring some basic deep learning techniques to denoise speech,
using the [PyTorch](https://pytorch.org/) framework.

:warning: PLEASE CHECKOUT BRANCH `dev` FOR WORKING RESULTS.
Results on this branch `master` are not satisfactory, and a slightly
different version with regards to the dataset creation, and pre-processing.

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

The dataset used in this project is the [LibriSpeech](https://www.openslr.org/12/) dataset,
as provided by [torchaudio](https://pytorch.org/audio/stable/_modules/torchaudio/datasets/librispeech.html),
which will take care of downloading and loading the dataset.

However, this dataset is not used as is, but it is processed to add artificial noise.
The entire dataset creation can be performed by running the utility script `create_dataset.py`.
Most experiments were run with the following setup:
```bash
# Create training+validation set
python create_dataset.py --start 0 --end 150 -o data/BabbledLibri/train
# Create training+validation set
python create_dataset.py --test --start 150 --end 160 -o data/BabbledLibri/test
```

This script contains some defaults for creating a small noised dataset from `N=20` LibriSpeech
utterances, which corresponds to a total of 1990 training samples, or around 33 minutes of noised
speech.

### Training

The training script takes in a configuration file in JSON format containing data, model, and
training parameters. Example configurations for the three types of available models are
available in `configs/`. To run a basic fully-connected autoencoder, run:
```python
python train.py --config configs/fcae.json
```
