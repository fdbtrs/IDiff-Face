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
    'eer': "#EC6500"
}

model_names = {
    "unet-cond-ca-bs512-150K": "ca",
    "unet-cond-ca-bs512-150K-cpd25": "ca",
    "unet-cond-ca-bs512-150K-cpd50": "ca",
}

model_titles = {
    "unet-cond-adagn-bs512-150K": "AdaGN",
    "unet-cond-adagn-bs512-150K-cpd25": "AdaGN-CPD25",
    "unet-cond-adagn-bs512-150K-cpd50": "AdaGN-CPD50",
    "unet-cond-ca-bs512-150K": "CrossAttention",
    "unet-cond-ca-bs512-150K-cpd25": "CrossAttention-CPD25",
    "unet-cond-ca-bs512-150K-cpd50": "CrossAttention-CPD50",
}

frm_titles = {
    "elasticface": "ElasticFace",
    "curricularface": "CurricularFace",
    "sface": "SFace",
    "synface": "SynFace",
    "usynthface": "USynthFace",
    "idiff-face": "IDiff-Face",
}

#AUTHENTIC_CONTEXTS_NAME = "random_elasticface_lfw_5000"
#SYNTHETIC_CONTEXTS_NAME = "random_synthetic_uniform_5000"


def read_scores_from_txt(f):
    scores = []
    for line in f.readlines():
        scores.append(float(line))
    return scores

def load_comparison_score_data(model_names, args):
    # read comparison score data
    model_scores = {}
    for model_name in model_names:

        model_scores[model_name] = {frm_name: {} for frm_name in args.frm_names}
        for frm_name in args.frm_names:

            path = os.path.join("evaluation", "lfw_aligned", model_name, frm_name)

            with open(os.path.join(path, "real_vs_real_genuine_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["real_vs_real_genuine"] = read_scores_from_txt(f)
            with open(os.path.join(path, "real_vs_real_imposter_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["real_vs_real_imposter"] = read_scores_from_txt(f)

            with open(os.path.join(path, "real_vs_variation_genuine_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["real_vs_variation_genuine"] = read_scores_from_txt(f)
            with open(os.path.join(path, "real_vs_variation_imposter_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["real_vs_variation_imposter"] = read_scores_from_txt(f)

            with open(os.path.join(path, "variation_vs_real_genuine_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["variation_vs_real_genuine"] = read_scores_from_txt(f)
            with open(os.path.join(path, "variation_vs_real_imposter_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["variation_vs_real_imposter"] = read_scores_from_txt(f)

            with open(os.path.join(path, "variation_vs_variation_genuine_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["variation_vs_variation_genuine"] = read_scores_from_txt(f)
            with open(os.path.join(path, "variation_vs_variation_imposter_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["variation_vs_variation_imposter"] = read_scores_from_txt(f)

            with open(os.path.join(path, "real_vs_reference_variation_genuine_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["real_vs_reference_variation_genuine"] = read_scores_from_txt(f)
            with open(os.path.join(path, "real_vs_reference_variation_imposter_scores.txt"), "r") as f:
                model_scores[model_name][frm_name]["real_vs_reference_variation_imposter"] = read_scores_from_txt(f)

    # create pandas dataframes for comparison data
    model_stats = {}
    model_dfs = {}

    for model_name in model_names:
        model_stats[model_name] = {}
        model_dfs[model_name] = {}

        for frm_name in args.frm_names:

            model_stats[model_name][frm_name] = {
                "real_vs_real": {},
                "real_vs_variation": {},
                "variation_vs_real": {},
                "variation_vs_variation": {},
                "variation_vs_real_both": {},
            }
            model_dfs[model_name][frm_name] = {
                "real_vs_real": None,
                "real_vs_variation": None,
                "variation_vs_real": None,
                "variation_vs_variation": None,
                "variation_vs_real_both": None,
            }

            # build data for plots
            for plot_type in [
                "real_vs_real", "real_vs_variation", "variation_vs_real", "variation_vs_variation", "real_vs_reference_variation"
            ]:

                genuine_scores = model_scores[model_name][frm_name][f'{plot_type}_genuine']
                imposter_scores = model_scores[model_name][frm_name][f'{plot_type}_imposter']

                stats = get_eer_stats(genuine_scores, imposter_scores)
                model_stats[model_name][frm_name][plot_type] = stats

                df = pd.DataFrame()
                df['scores'] = genuine_scores + imposter_scores
                df['label'] = ['genuine'] * len(genuine_scores) + ['imposter'] * len(imposter_scores)

                model_dfs[model_name][frm_name][plot_type] = df

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

        if plot_type in model_stats[model_name]:
            ax.axvline(x=model_stats[model_name][frm_name][plot_type].eer_th, c=TU_DESIGN_COLORS['eer'])

        if legend:
            labels = ['genuine', 'imposter'] if plot_type != "real_vs_synthetic" else ['random']
            handles = [mpatches.Patch(color=TU_DESIGN_COLORS[label], label=label) for label in labels]
            ax.legend(handles=handles, labels=labels, loc="upper left", title="")
        else:
            ax.get_legend().remove()

        if labels:
            if plot_type == "real_vs_real":
                ax.set_title(f"Real LFW Dataset ({frm_titles[frm_name]})")
            else:
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
                save_path = ensure_path_join("additional", "lfw_experiments", "comparison_scores_distributions", f"{plot_type}_{frm_name}_{model_name}_distributions.png")

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
        #"auc",
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
    import math
    data = []
    data_idxs = []
    for frm_name in args.frm_names:
        for plot_type in args.plot_types:
            for model_name in model_names:
                stats = model_stats[model_name][frm_name][plot_type]._asdict()

                stats_data = {
                    metric_name: round(stats[metric_name], 5) for metric_name in interesting_pyeer_metrics
                }

                for metric_name in derived_metrics:
                    stats_data[metric_name] = round(derived_metrics[metric_name](stats), 5)

                data.append(stats_data)
                data_idxs.append(f"{frm_name}/{plot_type}/{model_name}")



    stats_df = pd.DataFrame(data, index=data_idxs)
    stats_df.to_html(ensure_path_join("additional", "lfw_experiments", "pyeer_report.html"))
    stats_df.to_latex(ensure_path_join("additional", "lfw_experiments", "pyeer_report.tex"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="utils/Elastic_R100_295672backbone.pth",
        help="model path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ffhq_128",
        help="path to data directory",
    )
    parser.add_argument(
        "--frm_names",
        nargs='+',
        default=["sface", "usynthface", "idiff-face", "synface", "elasticface", "curricularface"]
    )
    parser.add_argument(
        "--plot_types",
        nargs='+',
        default=["real_vs_real", "real_vs_reference_variation", "variation_vs_variation"]
    )
    args = parser.parse_args()
    main(args)
