import os
import argparse
from tqdm import tqdm
import sys

import torch
import torchvision.transforms as transforms
import torchvision

from PIL import Image
from sklearn.preprocessing import normalize

import numpy as np

from utils.iresnet import iresnet100
from utils.irse import IR_101

import torchvision

import inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, parent_dir)

import sys
sys.path.insert(0, 'idiff-face-iccv2023-code/')

from utils.iresnet import iresnet100, iresnet50
from utils.irse import IR_101
from utils.synface_resnet import LResNet50E_IR
from utils.moco import MoCo

def load_img_paths(datadir):
    """load num_imgs many FFHQ images"""
    img_files = sorted(os.listdir(datadir))
    return [os.path.join(datadir, f_name) for f_name in img_files if f_name.endswith(".jpg") or f_name.endswith(".png")]


class ImageInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir):
        self.img_paths = load_img_paths(datadir)
        print("Number of images:", len(self.img_paths))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def load_elasticface(device):
    print("loading ElasticFace model...")
    ckpt = torch.load(os.path.join("utils", "Elastic_R100_295672backbone.pth"), map_location=device)
    backbone = iresnet100(num_features=512).to(device)
    backbone.load_state_dict(ckpt)
    return backbone

def load_curricularface(device):
    print("loading CurricularFace model...")
    backbone = IR_101([112, 112]).to(device)
    ckpt = torch.load(os.path.join("utils", "CurricularFace_Backbone.pth"), map_location=device)
    backbone.load_state_dict(ckpt)
    return backbone


def main(args):
    device = torch.device(0)
    bs = 1

    print("Dataset:", args.data_dir)
    if args.frm_name == "elasticface":
        model = load_elasticface(device)
    elif args.frm_name == "curricularface":
        model = load_curricularface(device)
    elif args.frm_name == "imagenet":
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = torch.nn.Identity()
        model = model.to(device)

    elif args.frm_name == "idiff-face":
        model = iresnet50(num_features=512).to(device)
        ckpt = torch.load(os.path.join("utils", "54684backbone.pth"), map_location=device)
        model.load_state_dict(ckpt)

    elif args.frm_name == "sface":
        model = iresnet50(num_features=512).to(device)
        ckpt = torch.load(os.path.join("utils", "79232backbone.pth"), map_location=device)
        model.load_state_dict(ckpt)

    elif args.frm_name == "usynthface":
        model = iresnet50(num_features=512)

        model = MoCo(base_encoder=iresnet50, dim=512, K=32768).to(device)
        ckpt = torch.load(os.path.join("utils", "checkpoint_051.pth"), map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model = model.encoder_q

    elif args.frm_name == "synface":
        model = LResNet50E_IR([112, 96]).to(device)
        ckpt = torch.load(os.path.join("utils", "model_10k_50_idmix_9197.pth"), map_location=device)["state_dict"]
        model.load_state_dict(ckpt)

    else:
        raise NotImplementedError

    model.eval()
    dataset = ImageInferenceDataset(args.data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Starting encoding ...")
    content = {}
    for i, (img_batch, filename_batch) in tqdm(enumerate(loader), total=len(loader)):
        img_batch = torchvision.transforms.functional.resize(img_batch, 112)

        if args.frm_name == "synface":
            img_batch = torchvision.transforms.functional.center_crop(img_batch, [112, 96])

        img_batch = img_batch.to(device)
        emb_batch = model(img_batch)
        emb_batch = torch.nn.functional.normalize(emb_batch).detach().cpu().numpy()

        for emb, filename in zip(emb_batch, filename_batch):
            content[filename.split(".")[0]] = emb

    torch.save(content, os.path.join(args.out_dir, f"embeddings_{args.frm_name}_" + os.path.basename(args.data_dir) + ".npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--frm_name",
        type=str,
        default="sface",
        help="[elasticface, curricularface, imagenet, idiff-face, synface, usynthface, sface]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="idiff-face-iccv2023-code/data/ffhq_128",
        help="path to data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="idiff-face-iccv2023-code/data",
        help="directory to save embedding file to"
    )
    args = parser.parse_args()
    main(args)
