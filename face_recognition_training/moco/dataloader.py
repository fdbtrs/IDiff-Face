import os
import logging
from os.path import join as ojoin
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms as T
from moco.data_utils import (
    check_for_folder_structure,
    load_first_dfg_path,
    load_syn_mophing_paths,
    load_folder_morphing_paths,
    load_real_paths,
    load_syn_paths,
    load_img_and_lmark_paths,
    load_latent_paths,
    split_pos_neg_pairs,
    load_supervised_paths,
    load_latents,
)
from morphing.online_morphing import morph_two_imgs


class ImageDataset(Dataset):
    def __init__(
        self,
        datadir,
        transform,
        transform_k=None,
        num_imgs=0,
        aug_type="default",
        morphing=False,
        lmark_dir="",
        epochs=0,
    ):
        """Initializes image paths and preprocessing module."""
        self.is_folder_struct = check_for_folder_structure(datadir)
        aug_type = aug_type.lower()
        if self.is_folder_struct and morphing:
            self.img_paths, self.img_paths2 = load_folder_morphing_paths(
                datadir, lmark_dir, num_imgs
            )
        elif (
            aug_type == "gan_ra_4_16"
            or aug_type == "gan_hf"
            or not self.is_folder_struct
        ) and not morphing:
            self.img_paths = load_syn_paths(datadir, num_imgs)
        elif not self.is_folder_struct and morphing:
            self.img_paths, self.img_paths2 = load_syn_mophing_paths(
                datadir, lmark_dir, num_imgs
            )
        else:
            self.img_paths = load_real_paths(datadir, num_imgs)

        self.lmark_dir = lmark_dir
        self.aug_disco = aug_type in ["gan_ra_4_16", "gan_hf"]
        self.aug_disco_hf = aug_type in ["nogan_hf", "nogan_ra_4_16"]
        self.online_disco = aug_type == "online_disco"
        self.tmp_dir = datadir
        self.num_imgs = num_imgs

        self.morphing = morphing
        self.epochs = epochs
        self.resize = T.Resize(112)
        self.transform_q = transform
        self.transform_k = transform if transform_k is None else transform_k
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def disco_augmentation(self, index):
        person_path = self.img_paths[index]
        # simulate online augmentation
        imgs = sorted(os.listdir(person_path))[: self.epochs * 2]
        img_file1 = random.choice(imgs)
        img_file2 = random.choice(imgs)
        if self.aug_disco_hf:
            img_file1 = imgs[0]
            img_file2 = img_file1

        img1 = Image.open(ojoin(person_path, img_file1))
        img1 = img1.convert("RGB")
        img2 = Image.open(ojoin(person_path, img_file2))
        img2 = img2.convert("RGB")
        q = self.transform_q(img1)
        k = self.transform_k(img2)
        return [q, k], index

    def online_disco_aug(self, index):
        index = index * 2
        offset = index % 8
        file_idx = index - offset
        img_batch = np.load(os.path.join(self.tmp_dir, f"{file_idx}.npy"))
        img1 = Image.fromarray(img_batch[offset], "RGB")
        img2 = Image.fromarray(img_batch[offset + 1], "RGB")
        img1 = self.resize(img1)
        img2 = self.resize(img2)
        q = self.transform_q(img1)
        k = self.transform_k(img2)
        return [q, k], index

    def lmark_path(self, img_path):
        img_file = os.path.basename(img_path)
        lmark_file = "lmarks_" + os.path.splitext(img_file)[0] + ".txt"

        if self.is_folder_struct:
            id_dir = os.path.basename(os.path.dirname(img_path))
            lmark_path = ojoin(self.lmark_dir, id_dir, lmark_file)
        else:
            lmark_path = ojoin(self.lmark_dir, lmark_file)
        return lmark_path

    def shuffle_paths2(self):
        logging.info("Shuffle paths2...")
        random.shuffle(self.img_paths2)

    def load_new_lmark_paths(self):
        logging.info("Loading new landmark paths...")
        self.img_paths, self.img_paths2 = load_folder_morphing_paths(
            self.tmp_dir, self.lmark_dir, self.num_imgs
        )

    def for_morphing(self, index):
        img_path1 = self.img_paths[index]
        img_path2 = self.img_paths2[index]
        lmark_path1 = self.lmark_path(img_path1)
        lmark_path2 = self.lmark_path(img_path2)

        img1 = Image.open(img_path1)
        img1 = img1.convert("RGB")
        img2 = Image.open(img_path2)
        img2 = img2.convert("RGB")

        morphed, weight = morph_two_imgs(
            np.asarray(img1), np.asarray(img2), lmark_path1, lmark_path2
        )
        morphed = Image.fromarray(np.uint8(morphed))

        q1 = self.transform_q(img1)
        q2 = self.transform_q(img2)
        morph = self.transform_k(morphed)
        k = self.transform_k(morphed)
        q = torch.stack([q1, q2], dim=0)
        weights = torch.tensor([weight, 1 - weight])

        return [q, morph, k], weights

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        if self.morphing:
            return self.for_morphing(index)
        elif self.online_disco:
            return self.online_disco_aug(index)
        elif self.aug_disco or self.aug_disco_hf:
            return self.disco_augmentation(index)
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        q = self.transform_q(image)
        k = self.transform_k(image)
        return [q, k], index

    def __len__(self):
        """Returns the total number of font files."""
        if self.online_disco:
            return self.num_imgs
        return len(self.img_paths)


class DAMorphingDataset(Dataset):
    def __init__(
        self,
        datadir,
        lmark_dir,
        real_datadir,
        real_lmark_dir,
        num_imgs,
        num_real_imgs,
        transform_q,
        transform_k,
        aug_type,
        epochs,
    ):
        self.img_paths, self.lmark_paths = load_img_and_lmark_paths(
            datadir, lmark_dir, num_imgs
        )
        all_real_img_paths, all_real_lmark_paths = load_img_and_lmark_paths(
            real_datadir, real_lmark_dir, num_ids=0
        )
        # flatten real lists
        self.real_img_paths = [
            item for sublist in all_real_img_paths for item in sublist
        ]
        self.real_lmark_paths = [
            item for sublist in all_real_lmark_paths for item in sublist
        ]
        if num_real_imgs != 0:
            self.real_img_paths = self.real_img_paths[:num_real_imgs]
            self.real_lmark_paths = self.real_lmark_paths[:num_real_imgs]
        self.transform_q = transform_q
        self.transform_k = transform_k
        self.no_gan = "noGAN" in aug_type
        self.epochs = epochs

    def __getitem__(self, index):
        id_imgs = self.img_paths[index]
        # simulate online augmentation
        imgs = id_imgs[: self.epochs * 2]
        random_idx = 0
        if not self.no_gan:
            random_idx = random.randint(0, len(imgs) - 1)
        img_path1 = imgs[random_idx]
        lmark_path1 = self.lmark_paths[index][random_idx]

        real_idx = random.randint(0, len(self.real_img_paths) - 1)
        img_path2 = self.real_img_paths[real_idx]
        lmark_path2 = self.real_lmark_paths[real_idx]

        img1 = Image.open(img_path1)
        img1 = img1.convert("RGB")
        img2 = Image.open(img_path2)
        img2 = img2.convert("RGB")

        morphed, weight = morph_two_imgs(
            np.asarray(img1), np.asarray(img2), lmark_path1, lmark_path2
        )
        morphed = Image.fromarray(np.uint8(morphed))

        q1 = self.transform_q(img1)
        q2 = self.transform_q(img2)
        morph = self.transform_k(morphed)
        k = self.transform_k(morphed)
        q = torch.stack([q1, q2], dim=0)
        weights = torch.tensor([weight, 1 - weight])
        return [q, morph, k], weights

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


class InferenceDataset(Dataset):
    def __init__(self, datadir, transform, num_imgs=0, num_ids=0):
        """Initializes image paths and preprocessing module."""
        self.is_folder_struct = check_for_folder_structure(datadir)
        if self.is_folder_struct:
            self.img_paths = load_real_paths(
                datadir, num_imgs, num_classes=num_ids
            )  # load_first_dfg_path()
        else:
            self.img_paths = load_syn_paths(datadir, num_imgs)

        self.transform = transform
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        img = self.transform(image)
        return img, self.img_paths[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


class SupervisedDataset(Dataset):
    def __init__(self, datadir, transform, num_persons, num_imgs):
        """Similar to ImageDataset, but limit the number of persons and images per person"""
        self.img_paths, self.labels = load_supervised_paths(
            datadir, num_persons, num_imgs
        )
        self.transform = transform
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns with corresponding label."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        img = self.transform(image)
        return img, self.labels[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


class ConcantTwoDatasets(Dataset):
    def __init__(self, datadir, datadir2, transform, num_persons, num_imgs):
        """Loads two datasets similar to ImageDataset, but assigns individual labels to both datasets
        even when class folders are named the same and limit the number of persons and images per person"""
        img_paths1, labels1 = load_supervised_paths(datadir, num_persons, num_imgs)
        labels1 = np.array(labels1)
        img_paths2, labels2 = load_supervised_paths(datadir2, num_persons, num_imgs)
        labels2 = np.array(labels2)
        labels2 = labels2 + labels1[-1] + 1
        self.img_paths = img_paths1 + img_paths2
        self.labels = np.concatenate((labels1, labels2), axis=0)
        self.transform = transform
        logging.info(f"Loading {len(self.img_paths)} images...")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns with corresponding label."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        img = self.transform(image)
        return img, self.labels[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


class OnlineGANDataset(Dataset):
    def __init__(self, latent_path, num_imgs):
        from GAN_control.generate_imgs import load_latent_w

        self.latent_w = load_latent_w(latent_path)
        if num_imgs > 0:
            self.latent_w = self.latent_w[:num_imgs]

    def __getitem__(self, index):
        return self.latent_w[index]

    def __len__(self):
        return len(self.latent_w)


class LatsDataset(Dataset):
    def __init__(self, num_imgs, latent_dim=512, lat_path=None, seed=42):
        self.lat_dim = latent_dim
        if lat_path == "None":
            np.random.seed(seed)
            self.latents = np.random.randn(num_imgs, latent_dim)
            self.norm = False
            print("random latent generation")
        else:
            self.latents = load_latents(lat_path, num_imgs)
            self.norm = False
        logging.info(f"Create {len(self.latents)} latent representations")

    def __getitem__(self, index):
        latent_codes = self.latents[index]  # .reshape(-1, self.lat_dim)
        if self.norm:
            norm = np.linalg.norm(latent_codes, axis=0, keepdims=True)
            latent_codes = latent_codes / norm * np.sqrt(self.lat_dim)
        return latent_codes

    def __len__(self):
        return len(self.latents)
