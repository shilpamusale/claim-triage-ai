# claimtriageai/inference/cluster_predictor.py

import joblib
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN, approximate_predict
from sentence_transformers import SentenceTransformer
from typing import Union
from pathlib import Path


from claimtriageai.utils.logger import get_logger
from claimtriageai.configs.paths import (
    CLUSTER_MODEL_PATH,
    CLUSTERING_OUTPUT_PATH,
    INFERENCE_INPUT_PATH,
    REDUCER_MODEL_PATH,
    SENTENCE_TRANSFORMER_MODEL_NAME,
)
from claimtriageai.inference.loader import load_model
from claimtriageai.inference.preprocessor import preprocess_for_inference

logger = get_logger("cluster_predictor")

def predict_clusters(
    raw_data: pd.DataFrame, 
    reducer_path: str, 
    clusterer_path: str,
    sbert_model_name: str = "all-MiniLM-L6-v2"
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Predicts clusters for new data using pre-trained UMAP and HDBSCAN models.
    Returns the labels and a DataFrame with UMAP coordinates.
    """
    logger.info("Starting cluster prediction for new data.")
    
    # --- Step 1: Load all pre-trained models ---
    try:
        reducer = joblib.load(reducer_path)
        clusterer = joblib.load(clusterer_path)
        sbert_model = SentenceTransformer(sbert_model_name)
        # Load the transformers for the structured data
        _, target_encoder, numeric_transformer = load_model()
        logger.info("Successfully loaded all models and transformers.")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise

    # --- Step 2: Create feature matrix exactly like in training ---
    logger.info("Embedding denial reasons...")
    texts = raw_data['denial_reason'].fillna("").tolist()
    text_embeddings = sbert_model.encode(texts, show_progress_bar=False)

    logger.info("Preprocessing structured features for clustering...")
    structured_encoded = preprocess_for_inference(
        raw_input=raw_data,
        target_encoder=target_encoder,
        numeric_transformer=numeric_transformer,
    )
    
    # Combine text and structured features
    feature_matrix = np.hstack((text_embeddings, structured_encoded.to_numpy()))
    logger.info(f"Created final feature matrix with shape: {feature_matrix.shape}")
    
    # --- Step 3: Use loaded models to transform and predict ---
    logger.info("Applying UMAP transformation to new data...")
    reduced_embeddings = reducer.transform(feature_matrix)
    
    logger.info("Predicting clusters using HDBSCAN approximate_predict...")
    predicted_labels, _ = approximate_predict(clusterer, reduced_embeddings)
    
    logger.info(f"Prediction complete. Assigned {len(np.unique(predicted_labels))} unique clusters.")

    # Create a DataFrame for the coordinates and return both
    umap_coords_df = pd.DataFrame(reduced_embeddings[:, :2], columns=['umap_x', 'umap_y'], index=raw_data.index)
    
    return predicted_labels, umap_coords_df