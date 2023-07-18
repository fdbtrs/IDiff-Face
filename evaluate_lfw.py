import os
from itertools import product

import matplotlib.pyplot as plt
from typing import Any
import math

import hydra
import torch
import torchvision
from pytorch_lightning.lite import LightningLite

from PIL import Image

from omegaconf import OmegaConf, DictConfig
import numpy as np

from utils.helpers import ensure_path_join, normalize_to_neg_one_to_one

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
import sys

from utils.iresnet import iresnet100

sys.path.insert(0, 'idiff-face-iccv2023-code/')


class EvaluatorLite(LightningLite):
    def run(self, cfg) -> Any:
        self.seed_everything(cfg.evaluation.seed)

        # load pre-defined lfw comparison pairs and prepare them
        with open(cfg.evaluation.lfw_pairs_path) as f:
            lfw_pairs = [line.rstrip('\n').split('\t') for line in f][1:]

        def translate_lfw_fnames(id_name, i):
            return f"{id_name}_{i.zfill(4)}"

        lfw_pairs_folds = [lfw_pairs[i*600:(i+1)*600] for i in range(10)]

        for fold_idx in range(10):
            for j, genuine_pair in enumerate(lfw_pairs_folds[fold_idx][:300]):
                id_name, i1, i2 = genuine_pair
                lfw_pairs_folds[fold_idx][j] = (translate_lfw_fnames(id_name, i1), translate_lfw_fnames(id_name, i2))

            for j, imposter_pair in enumerate(lfw_pairs_folds[fold_idx][300:]):
                j = j + 300
                id_name1, i1, id_name2, i2 = imposter_pair
                lfw_pairs_folds[fold_idx][j] = (translate_lfw_fnames(id_name1, i1), translate_lfw_fnames(id_name2, i2))

        all_lfw_genuine_pairs = []
        all_lfw_imposter_pairs = []

        for fold_idx in range(10):
            all_lfw_genuine_pairs.extend(lfw_pairs_folds[fold_idx][:300])
            all_lfw_imposter_pairs.extend(lfw_pairs_folds[fold_idx][300:])

        # sanity checks
        for genuine_pair in all_lfw_genuine_pairs:
            assert genuine_pair[0].split("_")[0] == genuine_pair[1].split("_")[0]

        for model_name in cfg.evaluation.model_names:

            for frm_name in cfg.evaluation.frm_names:

                if cfg.evaluation.aligned:
                    eval_dir = ensure_path_join("evaluation", "lfw_aligned", model_name, frm_name)
                    variation_preencoded_data_dir = os.path.join("samples", "aligned", "embeddings", model_name,
                                                                 cfg.evaluation.variation_contexts_name, frm_name)
                else:
                    eval_dir = ensure_path_join("evaluation", "lfw", model_name, frm_name)
                    variation_preencoded_data_dir = os.path.join("samples", "embeddings", model_name,
                                                                 cfg.evaluation.variation_contexts_name, frm_name)

                variation_embeddings = torch.load(os.path.join(variation_preencoded_data_dir, "embeddings.npy"))
                variation_labels = torch.load(os.path.join(variation_preencoded_data_dir, "labels.npy"))

                # load real data embeddings
                real_contexts = cfg.evaluation.real_contexts.get(frm_name)
                print("Real Contexts:", real_contexts)

                real_embeddings_dict = torch.load(real_contexts.real_contexts_path if not cfg.evaluation.aligned else real_contexts.real_contexts_aligned_path)

                real_labels = list(real_embeddings_dict.keys())
                real_embeddings = [real_embeddings_dict[label] for label in real_labels]

                variation_embeddings_dict = {label: emb for label, emb in zip(variation_labels, variation_embeddings)}

                variation_labels = np.array(variation_labels)
                real_labels = np.array(real_labels)

                if cfg.evaluation.real_vs_real_comparison:
                    print("Starting REAL vs. REAL comparison ...")
                    genuine_scores, imposter_scores = [], []

                    for genuine_pair in all_lfw_genuine_pairs:

                        id_name1, id_name2 = genuine_pair
                        e1, e2 = real_embeddings_dict[id_name1], real_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)

                        if cos_sim < 0.2:
                            print(genuine_pair, cos_sim)
                            
                        genuine_scores.append(cos_sim)

                    for imposter_pair in all_lfw_imposter_pairs:

                        id_name1, id_name2 = imposter_pair

                        e1, e2 = real_embeddings_dict[id_name1], real_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        imposter_scores.append(cos_sim)

                    plt.clf()
                    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.1), label="genuine", color="green", alpha=0.5)
                    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.1), label="imposter", color="red", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.legend()
                    plt.savefig(ensure_path_join(eval_dir, "real_vs_real_distributions.png"), dpi=512)

                    genuine_file_path = os.path.join(eval_dir, f"real_vs_real_genuine_scores.txt")
                    imposter_file_path = os.path.join(eval_dir, f"real_vs_real_imposter_scores.txt")

                    with open(genuine_file_path, "w") as f:
                        for score in genuine_scores:
                            f.write(f"{score}\n")

                    with open(imposter_file_path, "w") as f:
                        for score in imposter_scores:
                            f.write(f"{score}\n")

                    del genuine_scores
                    del imposter_scores

                if cfg.evaluation.variation_vs_variation_comparison:
                    print("Starting VARIATION vs. VARIATION comparison ...")
                    genuine_scores, imposter_scores = [], []

                    for genuine_pair in all_lfw_genuine_pairs:

                        id_name1, id_name2 = genuine_pair
                        if id_name1 not in variation_embeddings_dict or id_name2 not in variation_embeddings_dict:
                            print(f"Skipping {genuine_pair}")
                            continue

                        e1, e2 = variation_embeddings_dict[id_name1], variation_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        genuine_scores.append(cos_sim)

                    for imposter_pair in all_lfw_imposter_pairs:

                        id_name1, id_name2 = imposter_pair
                        if id_name1 not in variation_embeddings_dict or id_name2 not in variation_embeddings_dict:
                            print(f"Skipping {imposter_pair}")
                            continue

                        e1, e2 = variation_embeddings_dict[id_name1], variation_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        imposter_scores.append(cos_sim)

                    plt.clf()
                    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.1), label="genuine", color="green", alpha=0.5)
                    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.1), label="imposter", color="red", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.legend()
                    plt.savefig(ensure_path_join(eval_dir, "variation_vs_variation_distributions.png"), dpi=512)

                    genuine_file_path = os.path.join(eval_dir, f"variation_vs_variation_genuine_scores.txt")
                    imposter_file_path = os.path.join(eval_dir, f"variation_vs_variation_imposter_scores.txt")

                    with open(genuine_file_path, "w") as f:
                        for score in genuine_scores:
                            f.write(f"{score}\n")

                    with open(imposter_file_path, "w") as f:
                        for score in imposter_scores:
                            f.write(f"{score}\n")

                    del genuine_scores
                    del imposter_scores

                if cfg.evaluation.real_vs_variation_comparison:
                    print("Starting REAL vs. VARIATION comparisons ...")
                    genuine_scores, imposter_scores = [], []

                    for genuine_pair in all_lfw_genuine_pairs:

                        id_name1, id_name2 = genuine_pair
                        if id_name2 not in variation_embeddings_dict:
                            print(f"Skipping {genuine_pair}")
                            continue

                        e1, e2 = real_embeddings_dict[id_name1], variation_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        genuine_scores.append(cos_sim)

                    for imposter_pair in all_lfw_imposter_pairs:

                        id_name1, id_name2 = imposter_pair
                        if id_name2 not in variation_embeddings_dict:
                            print(f"Skipping {imposter_pair}")
                            continue

                        e1, e2 = real_embeddings_dict[id_name1], variation_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        imposter_scores.append(cos_sim)

                    plt.clf()
                    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.1), label="genuine", color="green", alpha=0.5)
                    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.1), label="imposter", color="red", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.legend()
                    plt.savefig(ensure_path_join(eval_dir, "real_vs_variation_distributions.png"), dpi=512)

                    genuine_file_path = os.path.join(eval_dir, f"real_vs_variation_genuine_scores.txt")
                    imposter_file_path = os.path.join(eval_dir, f"real_vs_variation_imposter_scores.txt")

                    with open(genuine_file_path, "w") as f:
                        for score in genuine_scores:
                            f.write(f"{score}\n")

                    with open(imposter_file_path, "w") as f:
                        for score in imposter_scores:
                            f.write(f"{score}\n")

                    real_vs_variation_stats = get_eer_stats(genuine_scores, imposter_scores)
                    generate_eer_report([real_vs_variation_stats], ["real_vs_variation"], "pyeer_report.html")

                    del genuine_scores
                    del imposter_scores

                if cfg.evaluation.variation_vs_real_comparison:
                    print("Starting VARIATION vs. REAL comparisons ...")
                    genuine_scores, imposter_scores = [], []

                    for genuine_pair in all_lfw_genuine_pairs:

                        id_name1, id_name2 = genuine_pair
                        if id_name1 not in variation_embeddings_dict:
                            print(f"Skipping {genuine_pair}")
                            continue

                        e1, e2 = variation_embeddings_dict[id_name1], real_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        genuine_scores.append(cos_sim)

                    for imposter_pair in all_lfw_imposter_pairs:

                        id_name1, id_name2 = imposter_pair
                        if id_name1 not in variation_embeddings_dict:
                            print(f"Skipping {imposter_pair}")
                            continue

                        e1, e2 = variation_embeddings_dict[id_name1], real_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        imposter_scores.append(cos_sim)

                    plt.clf()
                    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.1), label="genuine", color="green", alpha=0.5)
                    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.1), label="imposter", color="red", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.legend()
                    plt.savefig(ensure_path_join(eval_dir, "variation_vs_real_distributions.png"), dpi=512)

                    genuine_file_path = os.path.join(eval_dir, f"variation_vs_real_genuine_scores.txt")
                    imposter_file_path = os.path.join(eval_dir, f"variation_vs_real_imposter_scores.txt")

                    with open(genuine_file_path, "w") as f:
                        for score in genuine_scores:
                            f.write(f"{score}\n")

                    with open(imposter_file_path, "w") as f:
                        for score in imposter_scores:
                            f.write(f"{score}\n")

                    del genuine_scores
                    del imposter_scores

                if cfg.evaluation.real_vs_reference_variation_comparison:
                    print("Starting REAL VS. REFERENCE VARIATION comparisons ...")
                    genuine_scores, imposter_scores = [], []

                    for genuine_pair in all_lfw_genuine_pairs:

                        id_name1, _ = genuine_pair
                        if id_name1 not in variation_embeddings_dict:
                            print(f"Skipping {genuine_pair}")
                            continue

                        e1, e2 = real_embeddings_dict[id_name1], variation_embeddings_dict[id_name1]
                        cos_sim = np.dot(e1, e2)
                        genuine_scores.append(cos_sim)

                    for imposter_pair in all_lfw_imposter_pairs:

                        id_name1, id_name2 = imposter_pair
                        if id_name2 not in variation_embeddings_dict:
                            print(f"Skipping {imposter_pair}")
                            continue

                        e1, e2 = real_embeddings_dict[id_name1], variation_embeddings_dict[id_name2]
                        cos_sim = np.dot(e1, e2)
                        imposter_scores.append(cos_sim)

                    plt.clf()
                    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.1), label="genuine", color="green", alpha=0.5)
                    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.1), label="imposter", color="red", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.legend()
                    plt.savefig(ensure_path_join(eval_dir, "real_vs_reference_variation_distributions.png"), dpi=512)

                    genuine_file_path = os.path.join(eval_dir, f"real_vs_reference_variation_genuine_scores.txt")
                    imposter_file_path = os.path.join(eval_dir, f"real_vs_reference_variation_imposter_scores.txt")

                    with open(genuine_file_path, "w") as f:
                        for score in genuine_scores:
                            f.write(f"{score}\n")

                    with open(imposter_file_path, "w") as f:
                        for score in imposter_scores:
                            f.write(f"{score}\n")

                    del genuine_scores
                    del imposter_scores

    @staticmethod
    def get_indices_for_lfw_pair(lfw_pair, labels_1, labels_2):

        img_id1, img_id2 = lfw_pair

        i = np.random.choice(np.where(labels_1 == img_id1)[0])
        j = np.random.choice(np.where(labels_2 == img_id2)[0])

        return i, j


@hydra.main(config_path='configs', config_name='evaluate_lfw_config', version_base=None)
def evaluate(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    evaluator = EvaluatorLite(devices="auto", accelerator="auto")
    evaluator.run(cfg)


if __name__ == "__main__":
    evaluate()

