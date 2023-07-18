import argparse
import os

import torch
import torchvision.transforms as transforms

from matplotlib import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
import pandas as pd
import torchvision
from PIL import Image

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report

from tqdm import tqdm

from create_nn_visualisation import load_elasticface, FFHQInferenceDataset
from utils.helpers import ensure_path_join

TU_DESIGN_COLORS = {
    'genuine': "#009D81",
    'imposter': "#0083CC",
    'random': "#721085", #"#FDCA00",
    'eer': "#EC6500",
    "vs. real": "#FDCA00"
}

model_names = {
    'unet-cond-adagn-bs512-150K': "adagn",
    "unet-cond-adagn-bs512-150K-cpd25": "adagn",
    "unet-cond-adagn-bs512-150K-cpd50": "adagn",

    "unet-cond-ca-bs512-150K": "ca",
    "unet-cond-ca-bs512-150K-cpd25": "ca",
    "unet-cond-ca-bs512-150K-cpd50": "ca"
}

model_titles = {
    "unet-cond-adagn-bs512-150K": "AdaptiveGroupNorm",
    "unet-cond-adagn-bs512-150K-cpd25": "AdaptiveGroupNorm-CPD25",
    "unet-cond-adagn-bs512-150K-cpd50": "AdaptiveGroupNorm-CPD50",
    "unet-cond-ca-bs512-150K": "CrossAttention",
    "unet-cond-ca-bs512-150K-cpd25": "CrossAttention-CPD25",
    "unet-cond-ca-bs512-150K-cpd50": "CrossAttention-CPD50",
}

frm_titles = {
    "elasticface": "ElasticFace",
    "curricularface": "CurricularFace"
}

AUTHENTIC_CONTEXTS_NAME = "random_elasticface_ffhq_5000"
SYNTHETIC_CONTEXTS_NAME = "random_synthetic_uniform_5000"

model_sample_identities = {
    'real_vs_variations': np.random.choice(5000, 3, replace=False),
    'real_vs_synthetic': np.random.choice(5000, 3, replace=False),
    'synthetic_vs_synthetic': np.random.choice(5000, 3, replace=False),
}


def read_scores_from_txt(f):
    scores = []
    for line in f.readlines():
        scores.append(float(line))
    return scores

def load_image(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        return torchvision.transforms.functional.to_tensor(img)

def split_identity_grid(samples_dir, id_name, image_size: int):
    img = load_image(os.path.join(samples_dir, id_name))

    nrows = img.shape[1] // image_size
    ncols = img.shape[2] // image_size

    id_images = []
    for r in range(nrows):
        for c in range(ncols):
            tile = img[:, r * image_size: (r + 1) * image_size, c * image_size: (c + 1) * image_size]
            id_images.append(tile)

    return id_images

def load_comparison_score_data(model_names, args):
    # read comparison score data
    model_scores = {}
    for model_name in model_names:

        model_scores[model_name] = {frm_name: {} for frm_name in args.frm_names}
        for frm_name in args.frm_names:

            path = os.path.join("evaluation", f"ffhq_aligned_{args.synthetic_type}", model_name, frm_name)

            real_vs_variations_genuine_file = os.path.join(path, "real_vs_variations_genuine_scores.txt")
            if os.path.isfile(real_vs_variations_genuine_file):
                with open(real_vs_variations_genuine_file, "r") as f:
                    model_scores[model_name][frm_name]["real_vs_variations_genuine"] = read_scores_from_txt(f)

            real_vs_variations_imposter_file = os.path.join(path, "real_vs_variations_imposter_scores.txt")
            if os.path.isfile(real_vs_variations_imposter_file):
                with open(real_vs_variations_imposter_file, "r") as f:
                    model_scores[model_name][frm_name]["real_vs_variations_imposter"] = read_scores_from_txt(f)

            real_nn_vs_synthetic_file = os.path.join(path, "real_nn_vs_synthetic_scores.txt")
            if os.path.isfile(real_nn_vs_synthetic_file):
                with open(real_nn_vs_synthetic_file, "r") as f:
                    model_scores[model_name][frm_name]["real_nn_vs_synthetic"] = read_scores_from_txt(f)

            with open(os.path.join(path, "real_vs_synthetic_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["real_vs_synthetic"] = read_scores_from_txt(f)

            with open(os.path.join(path, "synthetic_vs_synthetic_genuine_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["synthetic_vs_synthetic_genuine"] = read_scores_from_txt(f)

            with open(os.path.join(path, "synthetic_vs_synthetic_imposter_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["synthetic_vs_synthetic_imposter"] = read_scores_from_txt(f)

    # create pandas dataframes for comparison data
    model_stats = {}
    model_dfs = {}

    for model_name in model_names:
        model_stats[model_name] = {}
        model_dfs[model_name] = {}

        for frm_name in args.frm_names:

            model_stats[model_name][frm_name] = {"real_vs_variations": {}, "synthetic_vs_synthetic": {}}
            model_dfs[model_name][frm_name] = {"real_vs_variations": None, "real_vs_synthetic": None, "synthetic_vs_synthetic": None}

            # build data for REAL vs. VARIATIONS plots
            if 'real_vs_variations_genuine' in model_scores[model_name][frm_name]:
                real_vs_variations_genuine_scores = model_scores[model_name][frm_name]['real_vs_variations_genuine']
                real_vs_variations_imposter_scores = model_scores[model_name][frm_name]['real_vs_variations_imposter']
                
                real_vs_variations_stats = get_eer_stats(real_vs_variations_genuine_scores, real_vs_variations_imposter_scores)
                model_stats[model_name][frm_name]["real_vs_variations"] = real_vs_variations_stats

                df = pd.DataFrame()
                df['scores'] = real_vs_variations_genuine_scores + real_vs_variations_imposter_scores
                df['label'] = ['genuine'] * len(real_vs_variations_genuine_scores) + ['imposter'] * len(real_vs_variations_imposter_scores)

                model_dfs[model_name][frm_name]["real_vs_variations"] = df

            # build data for REAL vs. SYNTHETIC plots
            real_vs_synthetic_scores = model_scores[model_name][frm_name]['real_nn_vs_synthetic']

            df = pd.DataFrame()
            df['scores'] = real_vs_synthetic_scores
            df['label'] = ['random'] * len(real_vs_synthetic_scores)

            model_dfs[model_name][frm_name]["real_vs_synthetic"] = df

            # build data for SYNTHETIC vs. SYNTHETIC plots
            synthetic_vs_synthetic_genuine_scores = model_scores[model_name][frm_name]['synthetic_vs_synthetic_genuine']
            synthetic_vs_synthetic_imposter_scores = model_scores[model_name][frm_name]['synthetic_vs_synthetic_imposter']

            synthetic_vs_synthetic_stats = get_eer_stats(synthetic_vs_synthetic_genuine_scores, synthetic_vs_synthetic_imposter_scores)
            model_stats[model_name][frm_name]["synthetic_vs_synthetic"] = synthetic_vs_synthetic_stats

            df = pd.DataFrame()
            df['scores'] = synthetic_vs_synthetic_genuine_scores + synthetic_vs_synthetic_imposter_scores
            df['label'] = ['genuine'] * len(synthetic_vs_synthetic_genuine_scores) + ['imposter'] * len(synthetic_vs_synthetic_imposter_scores)

            model_dfs[model_name][frm_name]["synthetic_vs_synthetic"] = df

    return model_scores, model_dfs, model_stats


def main(args):

    # general plot configurations
    sns.set_theme(style="whitegrid")

    params = {'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.labelweight': 'bold',
              'axes.titlesize': 'x-large',
              'axes.titleweight': 'bold',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'
              }
    pylab.rcParams.update(params)

    # load data for all models
    _, model_dfs, model_stats = load_comparison_score_data(model_names, args=args)

    # create figures with only nice distribution plots first
    def plot_score_histogram(ax, model_name, frm_name, plot_type, legend=False, labels=True):
        sns.histplot(ax=ax,
                     data=model_dfs[model_name][frm_name][plot_type], x="scores",
                     hue="label", palette=TU_DESIGN_COLORS,
                     stat="probability", kde=True, bins=100, binrange=(-1, 1))

        if plot_type in model_stats[model_name][frm_name]:
            ax.axvline(x=model_stats[model_name][frm_name][plot_type].eer_th, c=TU_DESIGN_COLORS['eer'])

        if legend:
            if plot_type == "real_vs_synthetic":
                labels = ['random']
            elif plot_type == "synthetic_vs_synthetic":
                labels = ['genuine', 'imposter']
            else:
                labels = ['genuine', 'imposter']

            handles = [mpatches.Patch(color=TU_DESIGN_COLORS[label], label=label) for label in labels]
            ax.legend(handles=handles, labels=labels, loc="upper left", title="")
        else:
            ax.get_legend().remove()

        if labels:
            ax.set_title(f"{model_titles[model_name]} Model ({frm_titles[frm_name]})")
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Probability")
        else:
            ax.set_title("")
            ax.set_ylabel("")
            ax.set_xlabel("")

        if plot_type == "real_vs_synthetic":
            ax.set_ylim(0, 0.145)
        else:
            ax.set_ylim(0, 0.075)

    # apply the above function to create the distribution plots
    for model_name in model_names:
        for frm_name in args.frm_names:
            for plot_type in args.plot_types:
                save_path = ensure_path_join("additional", f"ffhq_experiments_{args.synthetic_type}", "comparison_scores_distributions", f"{plot_type}_{frm_name}_{model_name}_distributions.png")
                if os.path.isfile(save_path):
                    print(f"Plot {save_path} already exists! Skipping it ...")
                    continue

                fig = plt.figure(figsize=(8, 5))
                plot_score_histogram(plt.gca(), model_name, frm_name, plot_type, legend=True)
                plt.tight_layout()
                plt.savefig(save_path, dpi=256)
                plt.close(fig)

    # create pyeer report
    interesting_pyeer_metrics = [
        "eer",
        "auc",
        "fmr100",
        "fmr1000",
        "gmean",
        "gstd",
        "imean",
        "istd"
    ]

    derived_metrics = {
        "fdr": lambda stats: (stats["gmean"] - stats["imean"]) ** 2 / (stats["gstd"] ** 2 + stats["istd"] ** 2)
    }

    data = []
    data_idxs = []
    for model_name in model_names:
        for frm_name in args.frm_names:
            for plot_type in args.plot_types:
                if plot_type in ["real_vs_variations", "synthetic_vs_synthetic"]:
                    stats = model_stats[model_name][frm_name][plot_type]._asdict()
                    
                    stats_data = {
                        metric_name: stats[metric_name] for metric_name in interesting_pyeer_metrics
                    }

                    for metric_name in derived_metrics:
                        stats_data[metric_name] = derived_metrics[metric_name](stats)

                    data.append(stats_data)
                    data_idxs.append(f"{plot_type}/{frm_name}/{model_name}")

    stats_df = pd.DataFrame(data, index=data_idxs)
    stats_df.to_html(ensure_path_join("additional", f"ffhq_experiments_{args.synthetic_type}", "pyeer_report.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="utils/Elastic_R100_295672backbone.pth",
        help="model path",
    )
    parser.add_argument(
        "--synthetic_type",
        type=str,
        default="synthetic_uniform",
        help="[synthetic_uniform, synthetic_learned, synthetic_extracted]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ffhq_128",
        help="path to data directory",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="number of nearest neighbors per distance type",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=2,
        help="square root of number of images per block",
    )
    parser.add_argument(
        "--frm_names",
        nargs='+',
        default=["elasticface"]#, "curricularface"]
    )
    parser.add_argument(
        "--plot_types",
        nargs='+',
        default=["synthetic_vs_synthetic", "real_vs_variations"]
    )
    args = parser.parse_args()
    main(args)
