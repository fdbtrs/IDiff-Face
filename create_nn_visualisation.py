import os
import argparse
from tqdm import tqdm
import sys

from torchvision.utils import save_image, make_grid

import torch
import torchvision.transforms as transforms

from PIL import Image

from utils.iresnet import iresnet100

import torchvision

import inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, parent_dir)


def load_ffhq_paths(datadir, num_imgs=0):
    """load num_imgs many FFHQ images"""
    img_files = sorted(os.listdir(datadir))
    if num_imgs != 0:
        img_files = img_files[:num_imgs]
    return [os.path.join(datadir, f_name) for f_name in img_files]


class FFHQInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, num_imgs=0):

        """Initializes image paths and preprocessing module."""
        self.img_paths = load_ffhq_paths(datadir, num_imgs)
        print("Number of images:", len(self.img_paths))
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def load_elasticface(device, model_path):
    print("loading ResNet model...")
    backbone = iresnet100(num_features=512).to(device)
    ckpt = torch.load(model_path, map_location=device)
    backbone.load_state_dict(ckpt)
    return backbone


def main(args):
    device = torch.device(0)
    batch_size = 32

    print("Dataset:", args.data_dir)

    if args.model_type.lower() == "elasticface":
        model = load_elasticface(device, args.model_path)
    else:
        raise NotImplementedError

    model.eval()
    dataset = FFHQInferenceDataset(args.data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    testing_img_paths = os.listdir(args.testing_img_dir)
    testing_imgs = []
    for path in testing_img_paths:
        with open(os.path.join(args.testing_img_dir, path), "rb") as f:
            img = Image.open(f)
            img = dataset.transform(img)
            testing_imgs.append(img)

    testing_imgs = torch.stack(testing_imgs)

    def embed_images(img_batch):
        img_batch = torchvision.transforms.functional.resize(img_batch, 112)
        img_batch = img_batch.to(device)
        emb_batch = model(img_batch).detach()
        return torch.nn.functional.normalize(emb_batch)

    testing_img_embeddings = embed_images(testing_imgs)

    nn_pixel_distances = torch.ones((len(testing_imgs), len(dataset))) * 9999999
    nn_embedding_distances = torch.ones((len(testing_imgs), len(dataset))) * 9999999

    print(f"Starting FFHQ NN Search for {len(testing_imgs)} images")
    for i, (img_batch, filename_batch) in tqdm(enumerate(loader), total=len(loader)):

        for j, img in enumerate(img_batch):
            # L1 distance on the pixel-level
            pixel_distance = torch.square(testing_imgs - img).sum([1, 2, 3])
            for kk, d in enumerate(pixel_distance):
                nn_pixel_distances[kk, i * batch_size + j] = d

        img_batch = transforms.functional.normalize(img_batch, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        with torch.no_grad():
            emb_batch = embed_images(img_batch)
            for j, emb in enumerate(emb_batch):
                # L2 distance on the embedding-level
                embedding_distance = torch.abs(testing_img_embeddings - emb).sum([1])
                for kk, d in enumerate(embedding_distance):
                    nn_embedding_distances[kk, i * batch_size + j] = d

    k = args.k
    nn_pixel_distances_sorted_idxs = torch.argsort(nn_pixel_distances, dim=1)
    nn_pixel_distances_sorted_idxs = nn_pixel_distances_sorted_idxs[:, :k]

    nn_pixel_distance_imgs = torch.zeros((len(testing_imgs), k, *img_batch.shape[-3:]))
    for i in range(len(testing_imgs)):
        for j in range(k):
            img, _ = dataset[nn_pixel_distances_sorted_idxs[i, j]]
            nn_pixel_distance_imgs[i, j] = img

    nn_embedding_distances_sorted_idxs = torch.argsort(nn_embedding_distances, dim=1)
    nn_embedding_distances_sorted_idxs = nn_embedding_distances_sorted_idxs[:, :k]

    nn_embedding_distance_imgs = torch.zeros((len(testing_imgs), k, *img_batch.shape[-3:]))
    for i in range(len(testing_imgs)):
        for j in range(k):
            img, _ = dataset[nn_embedding_distances_sorted_idxs[i, j]]
            nn_embedding_distance_imgs[i, j] = img

    images = torch.zeros((len(testing_imgs), 1+k+k, *img_batch.shape[-3:]))
    images[:, 0] = testing_imgs
    images[:, 1:1+k] = nn_pixel_distance_imgs
    images[:, 1+k:] = nn_embedding_distance_imgs

    images = (images + 1.0) / 2.0

    images = images.reshape(-1, *img_batch.shape[-3:])

    grid = make_grid(images, nrow=k+k+1, pad_value=1.0)

    save_image(grid, "nn.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/utils/Elastic_R100_295672backbone.pth",
        help="model path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ElasticFace",
        help="ElasticFace",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/ffhq_128/",
        help="path to data directory",
    )
    parser.add_argument(
        "--testing_img_dir",
        type=str,
        default="/data/testing_images",
        help="path to testing images directory",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="number of nearest neighbors per distance type",
    )
    args = parser.parse_args()
    main(args)
