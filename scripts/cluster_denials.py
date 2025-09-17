# scripts/cluster_denials.py

import argparse
import json

import pandas as pd

from claimtriageai.clustering.cluster_summary import (
    attach_cluster_labels,
    generate_cluster_labels,
)
from claimtriageai.configs.paths import (
    CLUSTER_LABELS_JSON_PATH,
    CLUSTER_MODEL_PATH,
    CLUSTERING_OUTPUT_PATH,
    DENIAL_PREDICTION_OUTPUT_PATH,
    REDUCER_MODEL_PATH,
)
from claimtriageai.inference.cluster_predictor import predict_clusters
from claimtriageai.utils.logger import get_logger

logger = get_logger("cluster_inference")

# In scripts/cluster_denials.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign clusters to new claims using a pre-trained model."
    )
    # ... (all your argparse arguments remain the same) ...
    parser.add_argument(
        "--input",
        type=str,
        default=DENIAL_PREDICTION_OUTPUT_PATH,
        help="Path to input CSV containing claims to be clustered.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CLUSTERING_OUTPUT_PATH,
        help="Path to save the output CSV with cluster assignments.",
    )
    parser.add_argument(
        "--reducer-model",
        type=str,
        default=REDUCER_MODEL_PATH,
        help="Path to the pre-trained UMAP reducer model (.joblib).",
    )
    parser.add_argument(
        "--cluster-model",
        type=str,
        default=CLUSTER_MODEL_PATH,
        help="Path to the pre-trained HDBSCAN clusterer model (.joblib).",
    )
    parser.add_argument(
        "--label-json",
        type=str,
        default=CLUSTER_LABELS_JSON_PATH,
        help="Path to save the generated cluster label map (.json).",
    )
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    df = df[df["denial_prediction"] == 1].copy()

    if df.empty:
        logger.warning("No denied claims to cluster.")
        pd.DataFrame(
            columns=list(df.columns) + ["denial_cluster_id", "umap_x", "umap_y"]
        ).to_csv(args.output, index=False)
        exit(0)

    # --- FIX: Unpack both the labels and the coordinates ---
    predicted_labels, umap_coords_df = predict_clusters(
        raw_data=df, reducer_path=args.reducer_model, clusterer_path=args.cluster_model
    )

    # --- FIX: Add BOTH cluster IDs and coordinates to the DataFrame ---
    df["denial_cluster_id"] = predicted_labels
    df = pd.concat([df, umap_coords_df], axis=1)

    logger.info("Successfully assigned cluster IDs and UMAP coordinates to claims.")

    # --- The rest of the logic remains the same ---
    label_map = generate_cluster_labels(df, method="mode")
    logger.info(f"Generated labels for {len(label_map)} clusters: {label_map}")

    with open(args.label_json, "w") as f:
        json.dump({int(k): v for k, v in label_map.items()}, f, indent=2)
        logger.info(f"Saved cluster label map to {args.label_json}")

    labeled_df = attach_cluster_labels(df, label_map)
    labeled_df.to_csv(args.output, index=False)
    logger.info(f"Clustering complete. Final output saved to {args.output}")
