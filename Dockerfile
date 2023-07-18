FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN pip install pytorch-lightning
RUN pip install hydra-core
RUN pip install einops
RUN pip install facenet-pytorch
RUN pip install scikit-image
RUN pip install seaborn
RUN pip install pyeer


