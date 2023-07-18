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

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report

from utils.helpers import ensure_path_join, normalize_to_neg_one_to_one

import sys

from utils.iresnet import iresnet100

sys.path.insert(0, 'idiff-face-iccv2023-code/')


class EvaluatorLite(LightningLite):
    def run(self, cfg) -> Any:
        self.seed_everything(cfg.evaluation.seed)

        eer_stats = {}

        for model_name in cfg.evaluation.model_names:

            for frm_name in cfg.evaluation.frm_names:

                synthetic_context_short = cfg.evaluation.synthetic_contexts_name.replace("random_", "").replace("_5000", "")

                if cfg.evaluation.aligned:
                    eval_dir = ensure_path_join("evaluation", f"ffhq_aligned_{synthetic_context_short}", model_name, frm_name)
                    synthetic_preencoded_data_dir = os.path.join("samples", "aligned", "embeddings", model_name,
                                                                 cfg.evaluation.synthetic_contexts_name, frm_name)
                    authentic_preencoded_data_dir = os.path.join("samples", "aligned", "embeddings", model_name,
                                                                 cfg.evaluation.authentic_contexts_name, frm_name)
                else:
                    eval_dir = ensure_path_join("evaluation", f"ffhq_{synthetic_context_short}", model_name, frm_name)
                    synthetic_preencoded_data_dir = os.path.join("samples", "embeddings", model_name,
                                                                 cfg.evaluation.synthetic_contexts_name, frm_name)
                    authentic_preencoded_data_dir = os.path.join("samples", "embeddings", model_name,
                                                                 cfg.evaluation.authentic_contexts_name, frm_name)

                synthetic_embeddings = torch.load(os.path.join(synthetic_preencoded_data_dir, "embeddings.npy"))
                synthetic_labels = torch.load(os.path.join(synthetic_preencoded_data_dir, "labels.npy"))

                authentic_embeddings = torch.load(os.path.join(authentic_preencoded_data_dir, "embeddings.npy"))
                authentic_labels = torch.load(os.path.join(authentic_preencoded_data_dir, "labels.npy"))

                # load real data embeddings
                real_contexts = cfg.evaluation.real_contexts.get(frm_name)
                print("Real Contexts:", real_contexts)

                real_embeddings_dict = torch.load(real_contexts.real_contexts_path if not cfg.evaluation.aligned else real_contexts.real_contexts_aligned_path)

                real_labels = list(real_embeddings_dict.keys())
                real_embeddings = [real_embeddings_dict[label] for label in real_labels]

                if cfg.evaluation.authentic_real_comparison:
                    print("Starting REAL vs. VARIATIONS comparison ...")
                    genuine_scores, imposter_scores = [], []

                    #for i, j in self.generate_genuine_pairs(authentic_labels, real_labels):
                    #    e1, e2 = authentic_embeddings[i], real_embeddings[j]
                    #    cos_sim = np.dot(e1, e2)
                    #    genuine_scores.append(cos_sim)

                    #for i, j in self.generate_random_imposter_pairs(authentic_labels, real_labels, n=len(genuine_scores)):
                    #    # TODO: save imposter pairing
                    #    e1, e2 = authentic_embeddings[i], real_embeddings[j]
                    #    cos_sim = np.dot(e1, e2)
                    #    imposter_scores.append(cos_sim)

                    comparison_scores = []
                    for i, e1 in enumerate(synthetic_embeddings[::16][:1000]):
                        print(i)
                        class_sim = []
                        for e2 in real_embeddings:
                            cos_sim = np.dot(e1, e2)
                            class_sim.append(cos_sim)
                        comparison_scores.append(np.sort(np.array(class_sim))[-1])

                    #plt.clf()
                    #plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.1), label="genuine", color="green", alpha=0.5)
                    #plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.1), label="imposter", color="red", alpha=0.5)
                    #plt.hist(comparison_scores, bins=np.arange(-1, 1, 0.1), label="real nn", color="purple", alpha=0.5)
                    #plt.xlim(-1, 1)
                    #plt.legend()
                    #plt.savefig(ensure_path_join(eval_dir, "real_vs_variations_distributions.png"), dpi=512)

                    #genuine_file_path = os.path.join(eval_dir, f"real_vs_variations_genuine_scores.txt")
                    #imposter_file_path = os.path.join(eval_dir, f"real_vs_variations_imposter_scores.txt")
                    nn_file_path = os.path.join(eval_dir, f"real_nn_vs_synthetic_scores.txt")

                    #with open(genuine_file_path, "w") as f:
                    #    for score in genuine_scores:
                    #        f.write(f"{score}\n")

                    #with open(imposter_file_path, "w") as f:
                    #    for score in imposter_scores:
                    #        f.write(f"{score}\n")

                    with open(nn_file_path, "w") as f:
                        for score in comparison_scores:
                            f.write(f"{score}\n")

                    #real_vs_variations_eer_stats = get_eer_stats(genuine_scores, imposter_scores)
                    #eer_stats[f"{model_name}_{frm_name}_real_vs_variations"] = real_vs_variations_eer_stats#

 #                   real_nn_vs_synthetic_eer_stats = get_eer_stats(comparison_scores, imposter_scores)
#                    eer_stats[f"{model_name}_{frm_name}_real_nn_vs_synthetic"] = real_nn_vs_synthetic_eer_stats

                    #del genuine_scores
                    #del imposter_scores
                    del comparison_scores

                if cfg.evaluation.synthetic_real_comparison:
                    print("Starting REAL vs. SYNTHETIC comparisons ...")

                    comparison_scores = []
                    for i, j in self.generate_random_imposter_pairs(synthetic_labels, real_labels, n=1_000_000):
                        e1, e2 = synthetic_embeddings[i], real_embeddings[j]
                        cos_sim = np.dot(e1, e2)
                        comparison_scores.append(cos_sim)

                    #comparison_scores = []
                    #for i, e1 in enumerate(np.random.choice(synthetic_embeddings[::16], 1000, replace=False)):
                    #    print(i)
                    #    class_sim = []
                    #    for e2 in real_embeddings:
                    #        cos_sim = np.dot(e1, e2)
                    #        class_sim.append(cos_sim)
                    #   comparison_scores.append(np.sort(np.array(class_sim))[-1])

                    plt.clf()
                    plt.hist(comparison_scores, bins=np.arange(-1, 1, 0.1), color="orange", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.savefig(ensure_path_join(eval_dir, "real_vs_synthetic_distributions.png"), dpi=512)

                    scores_file_path = os.path.join(eval_dir, f"real_vs_synthetic_scores.txt")

                    with open(scores_file_path, "w") as f:
                        for score in comparison_scores:
                            f.write(f"{score}\n")

                    del comparison_scores

                    # simpler exhaustive comparison pairings on subset of synthetic images
                    comparison_scores = []
                    for i in range(100):
                        for j in range(len(real_labels)):
                            e1, e2 = synthetic_embeddings[i * 16], real_embeddings[j]
                            cos_sim = np.dot(e1, e2)
                            comparison_scores.append(cos_sim)

                    plt.clf()
                    plt.hist(comparison_scores, bins=np.arange(-1, 1, 0.1), color="orange", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.savefig(ensure_path_join(eval_dir, "real_vs_synthetic_special_distributions.png"), dpi=512)

                    scores_file_path = os.path.join(eval_dir, f"real_vs_synthetic_special_scores.txt")

                    with open(scores_file_path, "w") as f:
                        for score in comparison_scores:
                            f.write(f"{score}\n")

                    del comparison_scores

                if cfg.evaluation.synthetic_synthetic_comparison:
                    print("Starting SYNTHETIC vs. SYNTHETIC comparisons ...")
                    genuine_scores, imposter_scores, nn_scores = [], [], []

                    for i, j in self.generate_genuine_pairs(synthetic_labels, synthetic_labels):
                        e1, e2 = synthetic_embeddings[i], synthetic_embeddings[j]
                        cos_sim = np.dot(e1, e2)
                        genuine_scores.append(cos_sim)

                    for i, j in self.generate_random_imposter_pairs(synthetic_labels, synthetic_labels, n=len(genuine_scores)):
                        # TODO: save imposter pairing
                        e1, e2 = synthetic_embeddings[i], synthetic_embeddings[j]
                        cos_sim = np.dot(e1, e2)
                        imposter_scores.append(cos_sim)

                    plt.clf()
                    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.1), label="genuine", color="green", alpha=0.5)
                    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.1), label="imposter", color="red", alpha=0.5)
                    plt.xlim(-1, 1)
                    plt.legend()
                    plt.savefig(ensure_path_join(eval_dir, "synthetic_vs_synthetic_distributions.png"), dpi=512)

                    genuine_file_path = os.path.join(eval_dir, f"synthetic_vs_synthetic_genuine_scores.txt")
                    imposter_file_path = os.path.join(eval_dir, f"synthetic_vs_synthetic_imposter_scores.txt")

                    with open(genuine_file_path, "w") as f:
                        for score in genuine_scores:
                            f.write(f"{score}\n")

                    with open(imposter_file_path, "w") as f:
                        for score in imposter_scores:
                            f.write(f"{score}\n")

                    synthetic_vs_synthetic_eer_stats = get_eer_stats(genuine_scores, imposter_scores)

                    eer_stats[f"{model_name}_{frm_name}_synthetic_vs_synthetic"] = synthetic_vs_synthetic_eer_stats
                    del genuine_scores
                    del imposter_scores

        report_path = os.path.join("evaluation", f"ffhq_aligned_{synthetic_context_short}", "pyeer_report.html") if cfg.evaluation.aligned else os.path.join("evaluation", f"ffhq_{synthetic_context_short}", "pyeer_report.html")
        generate_eer_report(list(eer_stats.values()), list(eer_stats.keys()), report_path)

    @staticmethod
    def generate_genuine_pairs(labels_1, labels_2, n=None, same_idx_okay=False):
        cnt = 0
        labels_1 = np.array(labels_1)
        labels_2 = np.array(labels_2)

        for label_1 in np.unique(labels_1):

            if n is not None and cnt >= n:
                break

            idxs_1 = np.where(labels_1 == label_1)[0]
            idxs_2 = np.where(labels_2 == label_1)[0]

            if len(idxs_1) * len(idxs_2) == 0:
                continue

            for i, j in product(idxs_1, idxs_2):

                if not same_idx_okay and i == j:
                    continue

                if n is not None and cnt >= n:
                    break

                cnt += 1

                yield i, j

    @staticmethod
    def generate_random_imposter_pairs(labels_1, labels_2, n=1024, ensure_no_duplicates=False):
        labels_1 = np.array(labels_1)
        labels_2 = np.array(labels_2)
        seen = set()
        i_list = np.random.choice(list(range(len(labels_1))), size=n, replace=True)
        for i in i_list:
            j = np.random.choice(np.where(labels_2 != labels_1[i])[0])

            if ensure_no_duplicates:
                if (i, j) not in seen:
                    seen.add((i, j))
                else:
                    continue

            yield i, j


@hydra.main(config_path='configs', config_name='evaluate_config', version_base=None)
def evaluate(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    evaluator = EvaluatorLite(devices="auto", accelerator="auto")
    evaluator.run(cfg)


if __name__ == "__main__":
    evaluate()

