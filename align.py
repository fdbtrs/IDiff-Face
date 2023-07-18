import os
from typing import Any

import hydra
import torch
import torchvision
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from pytorch_lightning.lite import LightningLite
import math

from PIL import Image

from omegaconf import OmegaConf, DictConfig
import numpy as np

from utils.alignment.arcface import norm_crop
from utils.helpers import ensure_path_join

from tqdm import tqdm

import sys

from facenet_pytorch import MTCNN

sys.path.insert(0, 'idiff-face-iccv2023-code/')


def load_image_paths(datadir, skip_files=None):
    img_files = os.listdir(datadir)

    if skip_files is not None:
        N = len(img_files)
        img_files = list(filter(lambda fname: fname not in skip_files, img_files))
        print(f"Skipped {N - len(img_files)} out of {N} images due to skip_files of length {len(skip_files)}")

    return [os.path.join(datadir, f_name) for f_name in img_files if f_name.endswith(".png") or f_name.endswith(".jpg")]


class ImageSplitInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, skip_files=None, image_size=128):

        self.img_paths = load_image_paths(datadir, skip_files)
        print("Number of images:", len(self.img_paths))

        self.image_size = image_size

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = image.convert("RGB")

        image = torchvision.transforms.functional.to_tensor(image)

        nrows = image.shape[1] // self.image_size
        ncols = image.shape[2] // self.image_size

        id_images = []
        for r in range(nrows):
            for c in range(ncols):
                tile = image[:, r * self.image_size: (r + 1) * self.image_size, c * self.image_size: (c + 1) * self.image_size]
                id_images.append(tile)
                
        return torch.stack(id_images), os.path.basename(img_path)

    def __len__(self):
        return len(self.img_paths)


class AlignerLite(LightningLite):

    def run(self, cfg) -> Any:

        mtcnn = MTCNN(
            keep_all=True, min_face_size=1, post_process=False, device=f"cuda:{self.global_rank}"
        )
        self.setup(mtcnn)

        print(self.global_rank, mtcnn.device)

        print("STARTING PRE-ALIGNMENT PROCESS FOR REAL DATA")
        for data_name in cfg.align.real_data_names:
            samples_dir = os.path.join("data", data_name)

            aligned_samples_dir = os.path.join("data", data_name + "_aligned")

            if os.path.isdir(aligned_samples_dir):
                print(f"Aligned real samples for {data_name} already exist.")
            else:
                os.makedirs(aligned_samples_dir, exist_ok=True)

                image_size = 250 if "lfw" in data_name else 128

                print(f"Starting alignment for {data_name}.")
                skipped_images = self.align_images(mtcnn, samples_dir, aligned_samples_dir, image_size=image_size)

                with open(ensure_path_join("data", f"skipped_images_{data_name}_aligned.txt"), "w") as f:
                    for id_name in skipped_images:
                        f.write(f"{id_name}\n")

        print("STARTING PRE-ALIGNMENT PROCESS FOR SYNTHETIC SAMPLES")
        for model_name in cfg.align.model_names:
            for contexts_name in cfg.align.contexts_names:

                # synthetic samples directory
                samples_dir = os.path.join("samples", model_name, contexts_name)
                aligned_samples_dir = os.path.join("samples", "aligned", model_name, contexts_name)

                if not os.path.isdir(samples_dir):
                    print(f"Samples dir for {model_name} and context {contexts_name} does not exist! You have to sample first! Skipping this context!")
                    continue

                aligned_sample_images = None
                if os.path.isdir(aligned_samples_dir):

                    sample_images = [fname for fname in os.listdir(samples_dir) if fname.endswith(".png")]
                    aligned_sample_images = [fname for fname in os.listdir(aligned_samples_dir) if fname.endswith(".png")]

                    if len(sample_images) == len(aligned_sample_images):
                        print(f"Aligned samples for {model_name} and context {contexts_name} already exist.")
                        continue
                    else:
                        print(
                            f"Some samples are not yet aligned for {model_name} and context {contexts_name}. Missing {len(sample_images) - len(aligned_sample_images)}. Continue ...")

                    del sample_images

                os.makedirs(aligned_samples_dir, exist_ok=True)

                print(f"Starting alignment for {model_name} and context {contexts_name}.")
                skipped_images = self.align_images(mtcnn, samples_dir, aligned_sample_images, aligned_samples_dir)
                print(f"Finished alignment for {model_name} and context {contexts_name}.")
                print(f"Number of skipped images: {len(skipped_images)}")

                with open(ensure_path_join("samples", "aligned", model_name, f"skipped_images_{contexts_name}.txt"), "w") as f:
                    for id_name in skipped_images:
                        f.write(f"{id_name}\n")


    @staticmethod
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

    def align_images(self, mtcnn, samples_dir, skip_files, aligned_samples_dir, image_size=128):

        dataset = ImageSplitInferenceDataset(datadir=samples_dir, skip_files=skip_files, image_size=image_size)
        dataset.img_paths = self.split_across_devices(dataset.img_paths)

        skipped_ids = []

        pbar = tqdm(enumerate(iter(dataset)), total=len(dataset))
        pbar.set_description("Alignment")

        pbar = iter(pbar)

        done_looping = False
        while not done_looping:

            try:
                i, (id_images, id_name) = next(pbar)
                aligned_sample_path = ensure_path_join(aligned_samples_dir, id_name)

            except StopIteration:
                print("STOPPED ITERATION")
                done_looping = True
                continue

            except Exception as e:
                print(f"Probably image file has been corrupted. Skipping this one. Exception: {e}")
                continue

            skip_identity = False
            id_images = id_images * 255

            if type(id_images) == np.ndarray:
                id_images = torch.from_numpy(id_images)
                print(id_images.shape)
            else:
                id_images = id_images.permute(0, 2, 3, 1)

            try:
                boxes, _, landmarks = mtcnn.detect(id_images.to(mtcnn.device), landmarks=True)

            except Exception as e:
                print("Exception:", e)
                skipped_ids.append(id_name)

                if type(id_images) == np.ndarray:
                    id_images = torch.from_numpy(id_images)
                else:
                    id_images = id_images.permute(0, 3, 1, 2)

                id_images = transforms.functional.resize(id_images, 112)

                if id_images.mean() > 1:
                    id_images = id_images / 255

                grid = make_grid(id_images, nrow=5, padding=0)
                print(f"WARNING: Skipped alignment for {id_name} due to an exception, a simple resized copy will be saved instead.")
                save_image(grid, aligned_sample_path)
                continue

            id_images = id_images.detach().cpu().numpy()
            aligned_id_images = []

            for j, (img, box, landmark) in enumerate(zip(id_images, boxes, landmarks)):

                if landmark is None:
                    skipped_ids.append(id_name)
                    skip_identity = True
                    break

                # find the face that is closest to the center
                box = np.array(box)

                img_center = (image_size // 2, image_size // 2)
                box_centers = np.array(list(zip((box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2)))
                offsets = box_centers - img_center
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 1)
                box_order = np.argsort(offset_dist_squared)

                facial5points = landmark[box_order[0]]

                aligned_img = norm_crop(img, landmark=facial5points, image_size=112, createEvalDB=True)

                aligned_img_tensor = torch.from_numpy(aligned_img).permute(2, 0, 1) / 255.0
                aligned_id_images.append(aligned_img_tensor)

            if skip_identity:

                id_images = torch.from_numpy(id_images)
                id_images = id_images.permute(0, 3, 1, 2)

                id_images = transforms.functional.resize(id_images, 112)

                if id_images.mean() > 1:
                    id_images = id_images / 255

                grid = make_grid(id_images, nrow=5, padding=0)
                print(
                    f"WARNING: Skipped alignment for {id_name} due to None as the landmark for one of them, a simple resized copy will be saved instead.")
                save_image(grid, aligned_sample_path)
                continue

            aligned_id_images = torch.stack(aligned_id_images)
            grid = make_grid(aligned_id_images, nrow=5, padding=0)
            save_image(grid, ensure_path_join(aligned_samples_dir, id_name))

            # pbar.set_postfix({"skips": f"{100*len(skipped_ids) / (i+1): .2f}%"})

        del dataset

        return skipped_ids

    def split_across_devices(self, L):
        if type(L) is int:
            L = list(range(L))

        chunk_size = math.ceil(len(L) / self.world_size)
        L_per_device = [L[idx: idx + chunk_size] for idx in range(0, len(L), chunk_size)]
        while len(L_per_device) < self.world_size:
            L_per_device.append([])

        return L_per_device[self.global_rank]

@hydra.main(config_path='configs', config_name='align_config', version_base=None)
def align(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = AlignerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":
    align()
