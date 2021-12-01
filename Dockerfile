FROM pytorch/pytorch:latest

RUN conda install -y -c conda-forge tqdm librosa
RUN conda install -y -c pytorch torchaudio torchvision

WORKDIR /workspace
CMD [ "python" ]
