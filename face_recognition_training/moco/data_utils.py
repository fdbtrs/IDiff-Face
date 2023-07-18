import os
from os.path import join as ojoin
import numpy as np
import random


def check_for_folder_structure(datadir):
    """checks if datadir contains folders (like CASIA) or images (synthetic datasets)"""
    img_path = sorted(os.listdir(datadir))[0]
    img_path = ojoin(datadir, img_path)
    return os.path.isdir(img_path)


def load_syn_mophing_paths(datadir, lmark_dir, num_imgs=0):
    """load image paths to given landmark files when these are all in one folder,
    not in class folders.
    args:
        datadir: path to directory containing images
        lmark_dir: path to directory containing landmarks
        num_imgs: number of images that should be loaded
    return:
        first half image paths, second half image paths
    """
    img_files = sorted(os.listdir(lmark_dir))
    if num_imgs != 0:
        img_files = img_files[:num_imgs]
    img_files = [img.split("lmarks_")[1].split(".")[0] for img in img_files]
    img_paths = [os.path.join(datadir, f_name + ".png") for f_name in img_files]
    return img_paths[: int(len(img_paths) / 2)], img_paths[int(len(img_paths) / 2) :]


def load_folder_morphing_paths(datadir, lmark_dir, num_imgs=0):
    """load image paths to given landmark files when these are in class folder structure
    args:
        datadir: path to directory containing images
        lmark_dir: path to directory containing landmarks
        num_imgs: number of images that should be loaded
    return:
        first half image paths, second half image paths
    """
    img_paths = []
    id_folders = sorted(os.listdir(lmark_dir))
    if num_imgs != 0:
        id_folders = id_folders[:num_imgs]
    for id in id_folders:
        imgs = os.listdir(ojoin(lmark_dir, id))
        img = random.choice(imgs)
        img = img.split("lmarks_")[1].split(".")[0]
        img += ".jpg"
        img_paths.append(ojoin(datadir, id, img))
    return img_paths[: int(len(img_paths) / 2)], img_paths[int(len(img_paths) / 2) :]


def load_img_and_lmark_paths(datadir, lmark_dir, num_ids=0):
    """loads image and corresponding landmark paths
    args:
        datadir: directory of images
        lmark_dir: directory of corresponding landmarks
        num_ids: number of identities, 0: select all ids
    return:
        [[id0_img0, id0_img1,...], [id1_img0,...]], same structure with landmark paths
    """
    id_folders = sorted(os.listdir(lmark_dir))
    if num_ids != 0:
        id_folders = id_folders[:num_ids]
    img_paths = []
    lmark_paths = []
    for id in id_folders:
        # list all landmarks for each id
        lmarks = os.listdir(ojoin(lmark_dir, id))
        # create full path and concatenate
        lmark_paths.append([ojoin(lmark_dir, id, l) for l in lmarks])
        # get img filename out of landmark filename
        imgs = [lmark.split("lmarks_")[1].split(".")[0] + ".jpg" for lmark in lmarks]
        img_paths.append([ojoin(datadir, id, img) for img in imgs])
    return img_paths, lmark_paths


def load_first_dfg_path(path, num_imgs):
    """loads complete image path of first image for each DFG class
    args:
        path: path to class folders
        num_imgs: number of images == number of classes
    return:
        list of image paths
    """
    img_paths = []
    id_folders = sorted(os.listdir(path))
    if num_imgs != 0:
        id_folders = id_folders[:num_imgs]
    for id in id_folders:
        img = sorted(os.listdir(ojoin(path, id)))[0]
        img_paths.append(ojoin(path, id, img))
    return img_paths


def load_real_paths(datadir, num_imgs=0, num_classes=0):
    """loads complete real image paths
    args:
        datadir: path to image folders
        num_imgs: number of total images
        num_classes: number of classes that should be loaded
    return:
        list of image paths
    """
    img_paths = []
    id_folders = sorted(os.listdir(datadir))
    if num_classes != 0:
        id_folders = id_folders[:num_classes]
    for id in id_folders:
        img_files = sorted(os.listdir(ojoin(datadir, id)))
        img_paths += [os.path.join(datadir, id, f_name) for f_name in img_files]
    if num_imgs != 0:
        img_paths = img_paths[:num_imgs]
    return img_paths


def load_syn_paths(datadir, num_imgs=0, start_img=0):
    """loads first level paths, i.e. image folders for DFG that contain augmentation images
    args:
        datadir: path to image folder
        num_imgs: number of images / folders
        start_img: start image index
    return:
        list of image paths
    """
    img_files = sorted(os.listdir(datadir))
    if num_imgs != 0:
        img_files = img_files[start_img : start_img + num_imgs]
    return [os.path.join(datadir, f_name) for f_name in img_files]


def load_latent_paths(dir_path):
    """load latent paths
    args:
        dir_path: path to directory of latents
    return:
        list of latent paths [[lat0_id0, lat1_id0,..],[lat0_id1,...],...]"""
    dirs = os.listdir(dir_path)
    lat_paths = []
    for dir in dirs:
        id_path = ojoin(dir_path, dir)
        id_lats = [ojoin(id_path, filename) for filename in os.listdir(id_path)]
        lat_paths.append(id_lats)
    return lat_paths


def split_pos_neg_pairs(lat_paths, num_pairs, is_val=False):
    """splits latent paths in positive and negative pairs and creates corresponding labels
    args:
        lat_paths: list of latent paths
        num_pairs: maximal number of positive and negative pairs each
    return:
        data [[2 pos paths], [pos], ... [neg], [2 neg paths]], labels [1,1,...,0,0]"""
    pos, neg = [], []
    path_len = len(lat_paths) // 2
    start_i = 0
    end_i = path_len
    if is_val:
        start_i = path_len
        end_i = path_len * 2
    for i in range(start_i, end_i, 1):
        max_idx = max(len(lat_paths[i]), 4)
        for j in range(0, max_idx, 2):
            pos.append([lat_paths[i][j], lat_paths[i][j + 1]])
            neg_idx = (i + 2) % path_len
            neg.append([lat_paths[i][j], lat_paths[neg_idx][j]])
    pos = pos[:num_pairs] if num_pairs != 0 else pos
    neg = neg[:num_pairs] if num_pairs != 0 else neg
    pos_y = np.ones(len(pos))
    neg_y = np.zeros(len(neg))
    labels = np.concatenate([pos_y, neg_y])
    data = pos + neg
    return data, labels


def load_supervised_paths(datadir, num_ids, num_imgs):
    """load e.g. DFG images with folder structure as supervised dataset
    args:
        datadir: path to directory containing the images
        num_ids: number of identities (folders) that should be loaded
        num_imgs: number of images per identity that should be loaded
    return:
        list of image paths, corresponding list of labels
    """
    img_paths, labels = [], []
    id_folders = sorted(os.listdir(datadir))[:num_ids]
    for i, id in enumerate(id_folders):
        id_path = ojoin(datadir, id)
        img_files = sorted(os.listdir(id_path))[:num_imgs]
        img_paths += [ojoin(id_path, f_name) for f_name in img_files]

        labels += [int(i)] * len(img_files)

    return img_paths, labels


def load_latents(datadir, num_lats=0):
    """load numpy latents from directory
    args:
        datadir: path to latent folder
        num_lats: number of latents
    return:
        numpy array of latents
    """
    lat_files = sorted(os.listdir(datadir))
    if num_lats != 0:
        lat_files = lat_files[:num_lats]
    lats = []
    for lat_file in lat_files:
        lats.append(np.load(ojoin(datadir, lat_file)))
    return np.array(lats)
