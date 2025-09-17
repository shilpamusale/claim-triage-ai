# """
# predict_denials.py — CLI Script to Predict Claim Denials

# This script:
# - Loads a trained denial prediction model and transformers
# - Reads raw claim data from a CSV
# - Applies full preprocessing and prediction
# - Saves results to an output CSV

# Usage:
# python scripts/predict_denials.py \
#     --input data/sample_input_for_inference.csv \
#     --output output/predictions.csv

# Author: ClaimTriageAI Team
# """

# import argparse
# from pathlib import Path

# import pandas as pd
# import yaml

# from claimtriageai.configs.paths import (
#     DENIAL_PREDICTION_OUTPUT_PATH,
#     FEATURE_CONFIG_PATH,
#     INFERENCE_INPUT_PATH,
# )
# from claimtriageai.inference.loader import load_model
# from claimtriageai.inference.predictor import predict_claims
# from claimtriageai.utils.logger import get_logger
# from claimtriageai.utils.postprocessing import standardize_prediction_columns

# # Initialize Logging
# logger = get_logger("inference")


# # # ------------------------- Main Function -------------------------
# def main() -> None:
#     """
#     Main function to orchestrate the batch prediction process.
#     """
#     parser = argparse.ArgumentParser(
#         description="Predict claim denials from a raw input CSV file."
#     )
#     parser.add_argument(
#         "--input",
#         type=str,
#         required=True,
#         help="Path to the raw claim CSV file for inference.",
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         required=True,
#         help="Path to save the output CSV with predictions.",
#     )

#     args = parser.parse_args()

#     logger.info(f"Reading input file from: {args.input}")
#     df_input = pd.read_csv(args.input)

#     logger.info(f"Input data has {df_input.shape[0]} rows and {df_input.shape[1]} columns.")

#     # Call the core prediction logic from the predictor module.
#     # This function now handles loading models and all other logic internally.
#     logger.info("Running prediction pipeline...")
#     output_df = predict_claims(raw_data=df_input)

#     # Save the final, enriched DataFrame to the output path
#     logger.info(f"Saving final output to: {args.output}")
#     Path(args.output).parent.mkdir(parents=True, exist_ok=True)
#     output_df.to_csv(args.output, index=False)

#     logger.info("Inference completed successfully.")

# if __name__ == "__main__":
#     main()

#     df = pd.read_csv(DENIAL_PREDICTION_OUTPUT_PATH)
#     with open(FEATURE_CONFIG_PATH, "r") as f:
#         config = yaml.safe_load(f)

#     # Check if the right columns made it into inference stage
#     used_cols = set(df.columns) & set(config["features"])
#     print("Used columns from feature_config.yaml:", used_cols)

"""
CLI Tool: Run Denial Prediction

This script runs the denial prediction model on a given input CSV
and saves the results to an output CSV.

Usage:
    python scripts/predict_denials.py --input <path> --output <path>

Author: ClaimTriageAI Team (2025)
"""

import argparse
import pandas as pd
from claimtriageai.utils.logger import get_logger
from claimtriageai.configs.paths import (
    DENIAL_PREDICTION_OUTPUT_PATH,
    INFERENCE_INPUT_PATH,
)

# --- FIX: Import the necessary functions ---
from claimtriageai.inference.loader import load_model
from claimtriageai.inference.predictor import predict_claims

# Initialize Logger
logger = get_logger("inference")

def main() -> None:
    """Main function to orchestrate the prediction process."""
    parser = argparse.ArgumentParser(description="Run denial prediction on new claims.")
    parser.add_argument(
        "--input",
        default=INFERENCE_INPUT_PATH,
        help="Path to the input CSV file of new claims."
    )
    parser.add_argument(
        "--output",
        default=DENIAL_PREDICTION_OUTPUT_PATH,
        help="Path to save the output CSV with predictions."
    )
    args = parser.parse_args()

    try:
        logger.info("Reading input file...")
        df_input = pd.read_csv(args.input)
        logger.info(f"Input File data: ({df_input.shape[0]}, {df_input.shape[1]})")

        # --- FIX: Load the model and transformers ---
        logger.info("Loading model and transformers...")
        model, target_encoder, numeric_transformer = load_model()

        # --- FIX: Pass the loaded artifacts to the prediction function ---
        logger.info("Running predictions...")
        results = predict_claims(
            raw_data=df_input,
            model=model,
            target_encoder=target_encoder,
            numeric_transformer=numeric_transformer
        )

        # Combine original data with predictions
        df_output = pd.concat([df_input.reset_index(drop=True), pd.DataFrame(results)], axis=1)

        logger.info(f"Saving output to: {args.output}")
        df_output.to_csv(args.output, index=False)
        logger.info("Inference completed successfully.")

    except FileNotFoundError:
        logger.error(f"Input file not found at: {args.input}")
    except Exception as e:
        logger.exception(f"An error occurred during prediction: {e}")
        raise

if __name__ == "__main__":
    main()