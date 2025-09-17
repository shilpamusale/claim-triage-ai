import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from claimtriageai.configs.paths import CLUSTERING_OUTPUT_PATH


def plot_umap_clusters(input_csv: str, labels_path: str, output_path: str) -> None:
    """
    Generates and saves a decorative UMAP scatter plot with cluster centroids and improved styling.
    """
    df = pd.read_csv(input_csv)

    if "umap_x" not in df.columns or "umap_y" not in df.columns:
        raise ValueError("Missing `umap_x` and `umap_y` columns in input file.")
    if "denial_cluster_id" not in df.columns:
        raise ValueError("Missing `denial_cluster_id` column in input file.")

    with open(labels_path, 'r') as f:
        cluster_label_map = json.load(f)
    df['cluster_name'] = df['denial_cluster_id'].astype(str).map(cluster_label_map).fillna("Noise - Unclustered")

    # --- Style ---
    sns.set_style("whitegrid")
    sns.set_context("talk")

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#1e1e2f")
    ax.set_facecolor("#2c2c3c")

    # --- Scatter plot ---
    scatter = sns.scatterplot(
        x="umap_x",
        y="umap_y",
        hue="cluster_name",
        hue_order=sorted(df['cluster_name'].unique()),
        palette="tab20",
        data=df,
        alpha=0.85,
        s=90,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        legend="full"
    )

    # --- Cluster centroids for labels ---
    centroids = df.groupby("cluster_name")[["umap_x", "umap_y"]].mean()
    for cluster, (x, y) in centroids.iterrows():
        ax.text(x, y, cluster, fontsize=12, fontweight="bold", color="white",
                ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"))

    # --- Titles & Labels ---
    ax.set_title("UMAP Projection of Denial Clusters", fontsize=22, fontweight="bold", color="white", pad=20)
    ax.set_xlabel("UMAP Dimension 1", fontsize=14, color="white")
    ax.set_ylabel("UMAP Dimension 2", fontsize=14, color="white")

    ax.tick_params(colors="white", which="both")

    # --- Legend ---
    legend = ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.setp(legend.get_texts(), color="white")
    plt.setp(legend.get_title(), color="white", fontweight="bold")

    # --- Final touches ---
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved fancy UMAP cluster plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot UMAP clusters (fancy version).")
    parser.add_argument("--input", default=CLUSTERING_OUTPUT_PATH, help="Path to the input CSV file.")
    parser.add_argument("--labels", default="data/cluster_labels.json", help="Path to the JSON cluster labels.")
    parser.add_argument("--output", default="data/umap_cluster_plot_fancy.png", help="Path to save the plot.")
    args = parser.parse_args()

    plot_umap_clusters(args.input, args.labels, args.output)
