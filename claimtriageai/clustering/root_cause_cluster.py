"""
Module: clustering/root_cause_cluster.py
    — Root Cause Clustering for Denied Healthcare Claims

This module performs unsupervised clustering of
denied claims to uncover latent denial patterns,
combining both textual (denial reason and follow-up notes)
and structured features (e.g., CPT, payer).

Features:
- Text embedding using lightweight Sentence-BERT: all-MiniLM-L6-v2
- Optionally includes follow-up notes with denial reason (configurable)
- Optional concatenation of structured features (CPT, payer, etc.)
- Dimensionality reduction using UMAP
- Clustering using HDBSCAN (density-based, no predefined k)
- Cluster assignment and optional labeling
- Persistence of artifacts: cluster model, UMAP reducer, clustered CSV

Intended Use:
- Supports downstream workflow routing and denial analytics
- Enables periodic re-clustering to detect emerging payer behavior
- Human-in-the-loop labeling or post-hoc cluster interpretation

Inputs:
- DataFrame with 'denial_reason' column (mandatory)
- Optional 'followup_notes' column (if use_notes=True)
- Optional structured fields: 'payer', 'CPT', etc.

Outputs:
- clustered_claims.csv with 'claim_id', 'denial_cluster_id', etc.
- Saved models: cluster_model.pkl, umap_reducer.pkl

Author: ClaimTriageAI Team (2025)
"""

from pathlib import Path
from typing import Any, List, Optional, Union, cast

import joblib
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from claimtriageai.configs.paths import (
    CLUSTER_MODEL_PATH,
    CLUSTERING_OUTPUT_PATH,
    PROCESSED_DATA_PATH,
    REDUCER_MODEL_PATH,
    SENTENCE_TRANSFORMER_MODEL_NAME,
)
from claimtriageai.inference.loader import load_model
from claimtriageai.inference.preprocessor import preprocess_for_inference
from claimtriageai.utils.logger import get_logger

logger = get_logger("cluster")


def embed_denial_reasons(
    texts: List[str], model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME
) -> np.ndarray[Any, Any]:
    logger.info(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    logger.info("Encoding denial reasons into embeddings.")
    return cast(np.ndarray[Any, Any], model.encode(texts, show_progress_bar=True))


def reduce_dimensions(
    data: np.ndarray[Any, Any], n_components: int = 5, random_state: int = 42
) -> tuple[np.ndarray[Any, Any], Optional[UMAP]]:
    if data.shape[0] < 3:
        logger.warning("Too few rows to perform clustering. Returning input unchanged.")
        return data, None

    effective_components = min(n_components, data.shape[0] - 1)
    logger.info(f"Reducing dimensions to {effective_components} using UMAP")
    reducer = UMAP(
        n_components=effective_components,
        n_neighbors=10,
        # min_dist=0.5,
        random_state=random_state,
        metric="cosine",
        densmap=False,
    )
    reduced = reducer.fit_transform(data)
    return reduced, reducer


def cluster_embeddings(
    data: np.ndarray[Any, Any], min_cluster_size: int = 8
) -> tuple[np.ndarray[Any, Any], HDBSCAN]:
    logger.info(
        f"Clustering data with HDBSCAN (min_cluster_size={min_cluster_size})..."
    )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,  # Added for robustness
        prediction_data=True,
    )
    labels = clusterer.fit_predict(data)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"Detected {num_clusters} clusters.")

    return labels, clusterer


# def cluster_embeddings(
#     data: np.ndarray[Any, Any], min_cluster_size: int = 15
# ) -> tuple[np.ndarray[Any, Any], HDBSCAN]:
#     logger.info("Clustering data with HDBSCAN")
#     n_samples = data.shape[0]
#     min_cluster_size = min(min_cluster_size, max(2, n_samples // 3))
#     clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
#     labels = clusterer.fit_predict(data)
#     logger.info(f"Detected {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
#     return labels, clusterer


def attach_clusters(
    df: pd.DataFrame,
    labels: np.ndarray[Any, Any],
    output_path: Union[str, Path] = CLUSTERING_OUTPUT_PATH,
) -> pd.DataFrame:
    df_with_clusters = df.copy()
    df_with_clusters["denial_cluster_id"] = labels
    df_with_clusters.to_csv(output_path, index=False)
    logger.info(f"Saved clustered claims to: {output_path}")
    return df_with_clusters


def save_artifacts(
    clusterer: HDBSCAN,
    reducer: UMAP,
    cluster_model_path: Union[str, Path] = CLUSTER_MODEL_PATH,
    reducer_model_path: Union[str, Path] = REDUCER_MODEL_PATH,
) -> None:
    joblib.dump(clusterer, cluster_model_path)
    joblib.dump(reducer, reducer_model_path)
    logger.info(
        f"Saved clusterer to {cluster_model_path} and reducer to {reducer_model_path}"
    )


def run_clustering_pipeline(
    df: pd.DataFrame,
    text_col: str = "denial_reason",
    notes_col: Optional[str] = "followup_notes",
    use_notes: bool = True,
    id_col: str = "claim_id",
    model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME,
    output_path: Union[str, Path] = CLUSTERING_OUTPUT_PATH,
    cluster_model_path: Union[str, Path] = CLUSTER_MODEL_PATH,
    reducer_model_path: Union[str, Path] = REDUCER_MODEL_PATH,
) -> pd.DataFrame:
    logger.info("Starting root cause clustering pipeline.")
    df = df[df["denied"] == 1].copy()

    if use_notes and notes_col in df.columns:
        logger.info(
            "Concatenating denial_reason and followup_notes for richer embedding."
        )
        texts = (df[text_col].fillna("") + " " + df[notes_col].fillna("")).tolist()
    else:
        logger.info("Using only denial_reason for embedding.")
        texts = df[text_col].fillna("").tolist()

    text_embeddings = embed_denial_reasons(texts, model_name)

    _, target_encoder, numeric_transformer = load_model()
    logger.info("Preprocessing structured features.")
    structured_encoded = preprocess_for_inference(
        raw_input=df,
        target_encoder=target_encoder,
        numeric_transformer=numeric_transformer,
    )

    feature_matrix = np.hstack((text_embeddings, structured_encoded))
    reduced, reducer = reduce_dimensions(feature_matrix)

    # Attach UMAP 2D projection for plotting
    if reduced.shape[1] >= 2:
        df["umap_x"] = reduced[:, 0]
        df["umap_y"] = reduced[:, 1]

    labels, clusterer = cluster_embeddings(reduced)

    df_text_cols = [text_col]
    if use_notes and notes_col in df.columns and notes_col is not None:
        df_text_cols.append(notes_col)

    output_cols = [id_col] + df_text_cols + ["umap_x", "umap_y"]
    clustered_df = attach_clusters(df[output_cols], labels, output_path)
    save_artifacts(clusterer, reducer, cluster_model_path, reducer_model_path)

    return clustered_df


# if __name__ == "__main__":
#     df = pd.read_csv(INFERENCE_INPUT_PATH)
#     run_clustering_pipeline(df)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a clustering model on processed claims data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=PROCESSED_DATA_PATH,
        help="Path to the processed claims data for training.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    run_clustering_pipeline(df)
