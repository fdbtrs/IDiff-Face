import os
from typing import Any

import torch
import torchvision
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

from PIL import Image

import numpy as np

import sys


def split_identity_grid(samples_dir, id_name, image_size: int):

    with open(os.path.join(samples_dir, id_name), "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

    img = torchvision.transforms.functional.to_tensor(img)

    nrows = img.shape[1] // image_size
    ncols = img.shape[2] // image_size

    id_images = []
    for r in range(nrows):
        for c in range(ncols):
            tile = img[:, r * image_size: (r + 1) * image_size, c * image_size: (c + 1) * image_size]
            id_images.append(tile)

    return id_images


if __name__ == "__main__":

    block_datasets_dir = "samples/aligned"
    datasets_dir = "datasets"

    model_names = [
        #"unet-cond-ca-bs512-150K",
        "unet-cond-ca-bs512-150K-cpd25",
        #"unet-cond-ca-bs512-150K-cpd50"
    ]

    for model_name in model_names:
        model_data_path = os.path.join(block_datasets_dir, model_name)

        for dataset_name in os.listdir(model_data_path):

            dataset_path = os.path.join(model_data_path, dataset_name)

            for id_block_file in os.listdir(dataset_path):

                id_images = split_identity_grid(dataset_path, id_block_file, image_size=112)
                id_name = id_block_file.split["."][0]

                for i, img in enumerate(id_images):

                    save_dir = os.path.join(datasets_dir, model_name, dataset_name)
                    os.makedirs(save_dir, exist_ok=True)

                    save_image(img, os.path.join(save_dir, f"{id_name}_{i}.png"))
