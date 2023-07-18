import os
from typing import Callable

import torch
from PIL import Image
from torchvision.datasets import DatasetFolder


class PILImageLoader:

    def __call__(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class SamplesWithEmbeddingsFileDataset(torch.utils.data.Dataset):

    def __init__(self,
                 samples_root: str,
                 embeddings_file_path: str,
                 sample_file_ending: str = '.png',
                 sample_loader: Callable = None,
                 sample_transform: Callable = None
                 ):
        super(SamplesWithEmbeddingsFileDataset, self).__init__()

        self.sample_loader = sample_loader
        self.transform = sample_transform

        self.embeddings_file_path = embeddings_file_path
        self.samples = self.build_samples(
            embeddings_file_path=embeddings_file_path,
            sample_root=samples_root,
            sample_file_ending=sample_file_ending
        )

    @staticmethod
    def build_samples(embeddings_file_path: str, sample_root: str, sample_file_ending: str):
        content = torch.load(embeddings_file_path)
        samples = []
        for index, (identity, embedding_npy) in enumerate(content.items()):
            image_path = os.path.join(sample_root, identity + sample_file_ending)
            samples.append((image_path, embedding_npy))
        return samples

    def __getitem__(self, index: int):
        image_path, embedding_npy = self.samples[index]

        image = self.sample_loader(image_path)
        embedding = torch.from_numpy(embedding_npy)

        if self.transform is not None:
            image = self.transform(image)

        return image, embedding

    def __len__(self):
        return len(self.samples)
