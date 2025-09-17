import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from claimtriageai.configs.paths import CLUSTERING_OUTPUT_PATH


def plot_umap_clusters(input_csv: str, labels_path: str, output_path: str) -> None:
    """
    Generates and saves a UMAP scatter plot with meaningful cluster labels.

    Args:
        input_csv (str): Path to the clustered data CSV.
        labels_path (str): Path to the cluster labels JSON file.
        output_path (str): Path to save the output plot image.
    """
    df = pd.read_csv(input_csv)

    if "umap_x" not in df.columns or "umap_y" not in df.columns:
        raise ValueError("Missing `umap_x` and `umap_y` columns in input file.")

    if "denial_cluster_id" not in df.columns:
        raise ValueError("Missing `denial_cluster_id` column in input file.")

    # FIX: Load the cluster labels from the JSON file
    with open(labels_path, 'r') as f:
        cluster_label_map = json.load(f)
        
    # FIX: Map the numeric cluster IDs to their string names.
    # The JSON keys are strings ('-1', '0', etc.), so we must cast the ID column to string first.
    # We also handle the -1 cluster (noise) by giving it a specific name.
    df['cluster_name'] = df['denial_cluster_id'].astype(str).map(cluster_label_map).fillna("Noise - Unclustered")

    plt.figure(figsize=(12, 10))
    
    # FIX: Use the new 'cluster_name' column for the hue
    sns.scatterplot(
        x="umap_x",
        y="umap_y",
        hue="cluster_name",
        hue_order=sorted(df['cluster_name'].unique()), # Ensures consistent legend order
        palette="viridis", # A different palette can sometimes be clearer
        data=df,
        legend="full",
        alpha=0.8,
        s=50 # Slightly larger points
    )
    
    plt.title("UMAP Projection of Denial Clusters", fontsize=18, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # FIX: Update the legend title
    plt.legend(title="Cluster Name", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved UMAP cluster plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot UMAP clusters from denial data.")
    parser.add_argument(
        "--input", 
        default=CLUSTERING_OUTPUT_PATH, 
        help="Path to the input CSV file with UMAP coordinates and cluster IDs."
    )
    # FIX: Add a new argument for the labels file
    parser.add_argument(
        "--labels", 
        default="data/cluster_labels.json", 
        help="Path to the JSON file containing cluster label mappings."
    )
    parser.add_argument(
        "--output", 
        default="data/umap_cluster_plot.png", 
        help="Path to save the output plot image."
    )
    args = parser.parse_args()

    plot_umap_clusters(args.input, args.labels, args.output)