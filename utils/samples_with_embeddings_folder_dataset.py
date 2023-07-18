import os
from typing import Callable

import torch
from PIL import Image
from torchvision.datasets import DatasetFolder


class NumpyEmbeddingLoader:

    def __call__(self, path):
        return torch.from_numpy(torch.load(path))


class PILImageLoader:

    def __call__(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class SamplesWithEmbeddingsFolderDataset(DatasetFolder):

    def __init__(self,
                 samples_root: str,
                 embeddings_root,
                 sample_file_ending: str = '.png',
                 embedding_file_ending: str = '.npy',
                 embedding_loader: Callable = None,
                 sample_loader: Callable = None,
                 sample_transform: Callable = None
                 ):
        super(SamplesWithEmbeddingsFolderDataset, self).__init__(
            root=embeddings_root,
            loader=sample_loader,
            extensions=[embedding_file_ending],
            transform=sample_transform,
            target_transform=None,
            is_valid_file=None
        )

        self.embedding_loader = embedding_loader
        self.update_samples(sample_root=samples_root, sample_file_ending=sample_file_ending)

    def find_classes(self, directory):
        classes = [""]
        class_to_idx = {"": None}

        return classes, class_to_idx

    def update_samples(self, sample_root: str, sample_file_ending: str):
        for index in range(len(self.samples)):
            embedding_path, _ = self.samples[index]

            image_path = os.path.join(sample_root, os.path.basename(embedding_path).split('.')[0] + sample_file_ending)
            assert os.path.isfile(embedding_path)

            self.samples[index] = (image_path, embedding_path)

    def __getitem__(self, index: int):
        image_path, embedding_path = self.samples[index]

        image = self.loader(image_path)
        embedding = self.embedding_loader(embedding_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, embedding
