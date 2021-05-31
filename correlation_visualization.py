import os
import torch
import matplotlib
import argparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from data_utils import load_datasets, load_chinese_tsv_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The model path to visualization")
    args = parser.parse_args()
    return args

def draw_figure(sims, labels, model_path, to_figures_dir=False):
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = plt.axes((.085, .15, .905, .84))

    points = ax.scatter(sims, labels, label='bert-base-nli', s=0.5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('similarity', fontsize=12)
    plt.ylabel('label', fontsize=12)
    plt.xticks([-1, 1], fontsize=10)
    plt.yticks([0, 1], fontsize=10)
#     ax.set_xticklabels(labels)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-0.1, 1.1)
#     ax.legend(fontsize=9, markerscale=1.0, loc=0)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)

    fig.tight_layout()

    # save to files in both png and pdf format
    from matplotlib.backends.backend_pdf import PdfPages
    if to_figures_dir:
        plt.savefig(f"./figures/corr_{os.path.basename(model_path.rstrip('/'))}.png", format="png")
    plt.savefig(os.path.join(model_path, "stsb_corr.png"), format="png")
    if to_figures_dir:
        with PdfPages(f"./figures/corr_{os.path.basename(model_path.rstrip('/'))}.pdf") as pdf:
            plt.savefig(pdf, format="pdf")
    with PdfPages(os.path.join(model_path, "stsb_corr.pdf")) as pdf:
        plt.savefig(pdf, format="pdf")
    

def corr_visualization(model_path, chinese_dataset="none", to_figures_dir=False):
    stsb_samples = load_datasets(datasets=["stsb"], need_label=True, use_all_unsupervised_texts=False, no_pair=False)
    if chinese_dataset != "none":
        stsb_samples = load_chinese_tsv_data(chinese_dataset, "test")
    model = SentenceTransformer(model_path)
    all_texts = []
    for sample in stsb_samples:
        all_texts.extend(sample.texts)
    all_labels = [sample.label for sample in stsb_samples]
    all_reps = model.encode(all_texts)
    all_sims = []
    for idx in range(0, len(all_reps), 2):
        sim = util.pytorch_cos_sim(all_reps[idx], all_reps[idx + 1]).item()
        all_sims.append(sim)
    assert len(all_sims) == len(all_labels) == len(stsb_samples)
    print(f"similarity mean: {torch.tensor(all_sims).mean().item()}")
    print(f"similarity std: {torch.tensor(all_sims).std().item()}")
    print(f"similarity max: {max(all_sims)}")
    print(f"similarity min: {min(all_sims)}")
    print(f"labels mean: {torch.tensor(all_labels).mean().item()}")
    print(f"labels std: {torch.tensor(all_labels).std().item()}")
    print(f"labels max: {max(all_labels)}")
    print(f"labels min: {min(all_labels)}")
    
    draw_figure(all_sims, all_labels, model_path, to_figures_dir=to_figures_dir)

if __name__ == "__main__":
    args = parse_args()
    corr_visualization(args.model_path, to_figures_dir=True)