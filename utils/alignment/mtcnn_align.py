import os
import cv2
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision.utils import save_image

from arcface import norm_crop

from facenet_pytorch import MTCNN

from PIL import Image

import sys

sys.path.insert(0, '/igd-slbt-master-thesis/')

mtcnn = MTCNN(
    select_largest=True, min_face_size=1, post_process=False, device="cuda:0"
)


def load_image_paths(datadir, num_imgs=0):
    """load num_imgs many FFHQ images"""
    img_files = sorted(os.listdir(datadir))
    if num_imgs != 0:
        img_files = img_files[:num_imgs]
    return [os.path.join(datadir, f_name) for f_name in img_files]


class ImageInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, num_imgs=0):

        """Initializes image paths and preprocessing module."""
        self.img_paths = load_image_paths(datadir, num_imgs)
        print("Number of images:", len(self.img_paths))
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def detect_and_align_single_batch(img_batch, mtcnn):
    img_batch = img_batch.permute(0, 2, 3, 1)
    img_batch = (img_batch * 255)

    _, _, landmarks = mtcnn.detect(img_batch, landmarks=True)

    img_batch = img_batch.detach().cpu().numpy()

    skipped_im
    for img, img_name, landmark in zip(img_batch, img_names, landmarks):
        if landmark is None:
            skipped_imgs.append(img_name)
            continue

        facial5points = landmark[0]
        warped_face = norm_crop(img, landmark=facial5points, image_size=112, createEvalDB=evalDB)

        save_image(torch.from_numpy(warped_face).permute(2, 0, 1) / 255.0, os.path.join(out_folder, img_name))
        counter += 1

def align_images(in_folder, out_folder, batch_size, evalDB=False):
    """MTCNN alignment for all images in in_folder and save to out_folder
    args:
            in_folder: folder path with images
            out_folder: where to save the aligned images
            batch_size: batch size
            num_imgs: amount of images to align - 0: align all images
            evalDB: evaluation DB alignment
    """
    os.makedirs(out_folder, exist_ok=True)

    dataset = ImageInferenceDataset(datadir=in_folder)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2
    )

    skipped_imgs = []
    counter = 0
    landmark_statistics = []

    for img_batch, img_names in tqdm(dataloader):
        img_batch = img_batch.to("cuda:0")\

        img_batch = img_batch.permute(0, 2, 3, 1)
        img_batch = (img_batch * 255)

        _, _, landmarks = mtcnn.detect(img_batch, landmarks=True)

        img_batch = img_batch.detach().cpu().numpy()

        for img, img_name, landmark in zip(img_batch, img_names, landmarks):
            if landmark is None:
                skipped_imgs.append(img_name)
                continue
            facial5points = landmark[0]

            landmark_statistics.append(facial5points)
            warped_face = norm_crop(img, landmark=facial5points, image_size=112, createEvalDB=evalDB)

            save_image(torch.from_numpy(warped_face).permute(2, 0, 1) / 255.0, os.path.join(out_folder, img_name))
            counter += 1

        print(np.mean(landmark_statistics, axis=0), np.std(landmark_statistics, axis=0))

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")


def main():
    parser = argparse.ArgumentParser(description="MTCNN alignment")
    parser.add_argument(
        "--in_folder",
        type=str,
        default="/workspace/igd-slbt-master-thesis/data/ffhq_128",
        help="folder with images",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="/workspace/igd-slbt-master-thesis/aligned",
        help="folder to save aligned images",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--evalDB", type=int, default=1, help="1 for eval DB alignment")

    args = parser.parse_args()
    align_images(
        args.in_folder,
        args.out_folder,
        args.batch_size,
        evalDB=args.evalDB == 1,
    )


if __name__ == "__main__":
    main()