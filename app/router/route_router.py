# """
# Route Router

# Handles the routing endpoint to score claims and assign A/R work queues.

# Features:
# - Accepts CSV uploads of claims
# - Routes each claim with a priority score and recommended queue
# - Returns list of routed claim dicts

# Author: ClaimTriageAI Team
# """

# from io import BytesIO
# from typing import Any, Dict, List, cast

# import numpy as np
# import pandas as pd
# from fastapi import APIRouter, File, HTTPException, UploadFile

# from claimtriageai.clustering.cluster_summary import (
#     attach_cluster_labels,
#     generate_cluster_labels,
# )
# from claimtriageai.clustering.clustering_pipeline import cluster_claims
# from claimtriageai.inference.loader import load_model
# from claimtriageai.inference.predictor import predict_claims
# from claimtriageai.routing.policy import PolicyEngine
# from claimtriageai.utils.logger import get_logger
# from claimtriageai.utils.postprocessing import standardize_prediction_columns

# # Initialize logger
# logger = get_logger("routing")

# router: APIRouter = APIRouter()
# policy_engine = PolicyEngine()


# @router.post("/route", response_model=List[Dict[str, Any]])  # type: ignore[misc]
# async def route_claims(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
#     """
#     Accepts a CSV file with claims and returns routed priorities.
#     """
#     try:
#         content = await file.read()
#         df = pd.read_csv(BytesIO(content))
#         routed = policy_engine.route_all(df)
#         return cast(List[Dict[str, Any]], routed.to_dict(orient="records"))
#     except Exception as e:
#         logger.exception("Routing failed.")
#         raise HTTPException(status_code=500, detail=f"Routing failed: {e}")


# @router.post("/fullroute", response_model=List[Dict[str, Any]])  # type: ignore[misc]
# async def full_route_claims(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
#     """
#     Accepts a CSV file, performs:
#     - denial prediction
#     - root cause clustering
#     - routing priority scoring

#     Returns:
#         List of routed claims as dicts.
#     """
#     try:
#         logger.info("Reading uploaded CSV...")
#         content = await file.read()
#         df_raw = pd.read_csv(BytesIO(content))

#         # Step 1: Denial Prediction
#         logger.info("Step 1: Denial Prediction")
#         model, target_encoder, numeric_transformer = load_model()
#         pred_results = predict_claims(
#             raw_data=df_raw,
#             model=model,
#             target_encoder=target_encoder,
#             numeric_transformer=numeric_transformer,
#         )

#         df_pred = pd.concat(
#             [df_raw.reset_index(drop=True), pd.DataFrame(pred_results)], axis=1
#         )
#         df_pred = standardize_prediction_columns(df_pred)

#         # Step 2: Root cause clustering
#         logger.info("Step 2: Root cause clustering")
#         clustered_df = cluster_claims(df_pred)
#         label_map = generate_cluster_labels(clustered_df)
#         labeled_df = attach_cluster_labels(clustered_df, label_map)

#         # Step 3: Routing
#         logger.info("Step 3: Routing")
#         policy_engine = PolicyEngine()
#         routed_df = policy_engine.route_all(labeled_df)

#         logger.info("Routing complete. Returning routed claim data.")
#         routed_df = routed_df.replace([np.inf, -np.inf], np.nan).fillna("NA")

#         return cast(List[Dict[str, Any]], routed_df.to_dict(orient="records"))

#     except Exception as e:
#         logger.exception("Full routing pipeline failed.")
#         raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

# In app/router/route_router.py

# --- Make sure these imports are at the top of the file ---


from io import BytesIO
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from claimtriageai.clustering.cluster_summary import (
    attach_cluster_labels,
    generate_cluster_labels,
)
from claimtriageai.configs.paths import CLUSTER_MODEL_PATH, REDUCER_MODEL_PATH
from claimtriageai.inference.cluster_predictor import predict_clusters
from claimtriageai.inference.loader import load_model
from claimtriageai.inference.predictor import predict_claims
from claimtriageai.routing.policy import PolicyEngine
from claimtriageai.utils.logger import get_logger

# Initialize logger
logger = get_logger("routing")
router: APIRouter = APIRouter()
policy_engine = PolicyEngine()


@router.post("/fullroute", response_model=List[Dict[str, Any]])  # type: ignore[misc]
async def full_route_claims(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """
    Accepts a CSV file, performs:
    - denial prediction
    - root cause clustering (on denied claims)
    - routing priority scoring
    """
    try:
        logger.info("--- Starting Full Triage Pipeline via API ---")
        content = await file.read()
        df_raw = pd.read_csv(BytesIO(content))

        # --- Step 1: Denial Prediction ---
        logger.info("Step 1: Running Denial Prediction...")
        model, target_encoder, numeric_transformer = load_model()
        pred_results = predict_claims(
            raw_data=df_raw,
            model=model,
            target_encoder=target_encoder,
            numeric_transformer=numeric_transformer,
        )

        # --- FIX: Drop the redundant 'claim_id'
        # from the results before concatenating ---
        pred_results_df = pd.DataFrame(pred_results).drop(columns=["claim_id"])
        df_pred = pd.concat([df_raw.reset_index(drop=True), pred_results_df], axis=1)

        # --- Step 2: Root Cause Clustering ---
        logger.info("Step 2: Running Root Cause Clustering...")
        denied_claims_df = df_pred[df_pred["denial_prediction"] == 1].copy()

        not_denied_claims_df = df_pred[df_pred["denial_prediction"] == 0].copy()
        not_denied_claims_df["denial_cluster_id"] = -1
        not_denied_claims_df["cluster_label"] = "Not Denied"
        not_denied_claims_df["umap_x"] = np.nan
        not_denied_claims_df["umap_y"] = np.nan

        labeled_df = pd.DataFrame()

        if not denied_claims_df.empty:
            # predicted_labels, umap_coords_df = predict_clusters(
            #     raw_data=denied_claims_df,
            #     reducer_path=REDUCER_MODEL_PATH,
            #     clusterer_path=CLUSTER_MODEL_PATH,
            # )
            predicted_labels, umap_coords_df = predict_clusters(
                raw_data=denied_claims_df,
                reducer_path=str(REDUCER_MODEL_PATH),
                clusterer_path=str(CLUSTER_MODEL_PATH),
                # sbert_model_name=SBERT_MODEL_NAME,
            )
            denied_claims_df["denial_cluster_id"] = predicted_labels
            denied_claims_df = pd.concat(
                [
                    denied_claims_df.reset_index(drop=True),
                    umap_coords_df.reset_index(drop=True),
                ],
                axis=1,
            )

            label_map = generate_cluster_labels(denied_claims_df)
            labeled_df = attach_cluster_labels(denied_claims_df, label_map)

        final_df = pd.concat([labeled_df, not_denied_claims_df], ignore_index=True)

        # --- Step 3: Routing ---
        logger.info("Step 3: Calculating Routing Priority...")
        policy_engine = PolicyEngine()
        routed_df = policy_engine.route_all(final_df)

        logger.info("--- Full Triage Pipeline Complete ---")
        routed_df = routed_df.replace([np.inf, -np.inf], np.nan).fillna("NA")

        return cast(List[Dict[str, Any]], routed_df.to_dict(orient="records"))

    except Exception as e:
        logger.exception("Full routing pipeline failed.")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
