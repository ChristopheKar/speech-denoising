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
The entire dataset creation can be performed by running `python dataset.py`, or interactively
within the corresponding Jupyter notebook `dataset.ipynb`.

### Training

If the environment has been correctly setup, then running `python train.py` should
train and evaluate a basic model.
