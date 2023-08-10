
"""
Process for evaluation:

Input: folder with samples from a given model

Output: distributions of genuine and imposter curves

"""

import os
from typing import Any

import hydra
import torch
import torchvision
from pytorch_lightning.lite import LightningLite

from PIL import Image

from omegaconf import OmegaConf, DictConfig
import numpy as np

from utils.helpers import ensure_path_join, normalize_to_neg_one_to_one

import sys

from utils.iresnet import iresnet100, iresnet50
from utils.irse import IR_101
from utils.synface_resnet import LResNet50E_IR
from utils.moco import MoCo

sys.path.insert(0, 'IDiff-Face/')


class EncoderLite(LightningLite):
    def run(self, cfg) -> Any:

        face_backbone = None

        for frm_name in cfg.encode.frm_names:
            print(f"Starting Encoding Process for FRM: {frm_name}")
            del face_backbone

            # instantiate face recognition backbone
            if frm_name == "elasticface":
                face_backbone = iresnet100(num_features=512)
                ckpt = torch.load(os.path.join("utils", "Elastic_R100_295672backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "curricularface":
                face_backbone = IR_101([112, 112])
                ckpt = torch.load(os.path.join("utils", "CurricularFace_Backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "idiff-face":
                face_backbone = iresnet50(num_features=512)
                ckpt = torch.load(os.path.join("utils", "54684backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "sface":
                face_backbone = iresnet50(num_features=512)
                ckpt = torch.load(os.path.join("utils", "79232backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "usynthface":
                face_backbone = iresnet50(num_features=512)

                face_backbone = MoCo(base_encoder=iresnet50, dim=512, K=32768)
                ckpt = torch.load(os.path.join("utils", "checkpoint_051.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt, strict=False)
                face_backbone = face_backbone.encoder_q

            elif frm_name == "synface":
                face_backbone = LResNet50E_IR([112, 96])
                ckpt = torch.load(os.path.join("utils", "model_10k_50_idmix_9197.pth"), map_location="cpu")["state_dict"]
                face_backbone.load_state_dict(ckpt)


            # push face recognition backbone to device
            face_backbone = self.setup(face_backbone)
            face_backbone.eval()

            for model_name in cfg.encode.model_names:

                for contexts_name in cfg.encode.contexts_names:

                    # build paths to directories
                    if cfg.encode.aligned:
                        samples_dir = ensure_path_join("samples", "aligned", model_name, contexts_name)
                        embeddings_dir = ensure_path_join("samples", "aligned", "embeddings", model_name, contexts_name, frm_name)
                    else:
                        samples_dir = ensure_path_join("samples", model_name, contexts_name)
                        embeddings_dir = ensure_path_join("samples", "embeddings", model_name, contexts_name, frm_name)

                    if not os.path.isdir(samples_dir):
                        print(f"Samples directory {samples_dir} does not exist! You have to sample first! Skipping this one!")
                        continue

                    if os.path.isfile(os.path.join(embeddings_dir, f"embeddings.npy")):
                        print(f"The directory {embeddings_dir} already exists. Skip contexts: {contexts_name}")
                        continue

                    # encode images with the given face recognition model
                    embeddings, labels = self.encode_images(face_backbone, samples_dir, image_size=112 if cfg.encode.aligned else cfg.encode.image_size)

                    # save authentic pre-encoded data
                    torch.save(embeddings, os.path.join(embeddings_dir, "embeddings.npy"))
                    torch.save(labels, os.path.join(embeddings_dir, "labels.npy"))

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
                tile = img[:, r * image_size: (r+1) * image_size, c * image_size: (c+1) * image_size]
                id_images.append(tile)

        return id_images

    def encode_images(self, face_backbone, samples_dir, image_size=128):
        embeddings = []
        id_labels = []
        for i, id_name in enumerate(sorted(os.listdir(samples_dir))):

            if not id_name.endswith(".png"):
                continue

            print("Encoding:", i, id_name)

            # maybe make this a PyTorch dataset
            id_images = self.split_identity_grid(samples_dir, id_name, image_size=image_size)
            id_images = torch.stack(id_images)
            id_images = normalize_to_neg_one_to_one(id_images)

            id_images = id_images.cuda()

            id_images = torchvision.transforms.functional.resize(id_images, 112)

            #id_images = torchvision.transforms.functional.center_crop(id_images, [112, 96])

            id_embeds = face_backbone(id_images)
            id_embeds = torch.nn.functional.normalize(id_embeds)

            for embed in id_embeds.detach().cpu().numpy():
                embeddings.append(embed)
                id_labels.append(id_name.split(".")[0])

        return np.array(embeddings), np.array(id_labels)


@hydra.main(config_path='configs', config_name='encode_config', version_base=None)
def encode(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = EncoderLite(devices=[0], accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":

    encode()

