"""
Module: predictor.py — Denial Prediction Inference Logic

This module implements the core business logic to:
- Load all necessary configurations and trained model artifacts.
- Accept a DataFrame of raw claims.
- Apply the full, consistent preprocessing pipeline.
- Generate predictions and merge them with the original data.

Author: ClaimTriageAI Team
"""

from typing import Any, Dict, List, Union, cast

import pandas as pd
import yaml
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from claimtriageai.configs import paths
from claimtriageai.inference.preprocessor import preprocess_for_inference
from claimtriageai.utils.logger import get_logger

# Initialize Logging
logger = get_logger("inference")


def load_and_merge_configs() -> dict[str, Any]:
    """
    Loads feature config from YAML and merges paths from the paths.py module
    to create a single, complete configuration object.
    """
    logger.info("Loading feature configuration from YAML...")
    with open(paths.FEATURE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading paths from paths.py module...")
    # The vars() function turns the paths.py module's attributes into a dictionary
    config["paths"] = vars(paths)
    return cast(dict[str, Any], config)


def predict_claims(
    raw_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model: BaseEstimator,
    target_encoder: TargetEncoder,
    numeric_transformer: ColumnTransformer,
) -> List[Dict[str, Any]]:
    """
    Generate denial predictions from raw claim data.

    Args:
        raw_data: Input data (dict, list of dicts, or DataFrame)
        model: Trained denial prediction model
        target_encoder: Fitted target encoder
        numeric_transformer: Fitted numeric/boolean transformer

    Returns:
        List of prediction dicts.
    """
    if isinstance(raw_data, dict):
        df_input = pd.DataFrame([raw_data])
    elif isinstance(raw_data, list):
        df_input = pd.DataFrame(raw_data)
    else:
        df_input = raw_data.copy()

    # Preprocess features
    logger.info("Preprocessing features for prediction...")
    processed_df = preprocess_for_inference(
        df_input, target_encoder, numeric_transformer
    )

    # Predict
    logger.info("Generating denial probabilities...")
    # Get probability of the positive class (denial) which is the second column
    probabilities = model.predict_proba(processed_df)[:, 1]

    results = []
    # Ensure claim_id is treated as a string to avoid JSON serialization issues
    df_input["claim_id"] = df_input["claim_id"].astype(str)

    for idx, row in df_input.iterrows():
        prob = float(probabilities[idx])
        result = {
            "claim_id": row.get("claim_id", "N/A"),
            "denial_probability": prob,
            "denial_prediction": 1 if prob > 0.5 else 0,
        }
        results.append(result)

    return results
