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

from claimtriageai.configs.paths import (
    DENIAL_PREDICTION_OUTPUT_PATH,
    INFERENCE_INPUT_PATH,
)

# --- FIX: Import the necessary functions ---
from claimtriageai.inference.loader import load_model
from claimtriageai.inference.predictor import predict_claims
from claimtriageai.utils.logger import get_logger

# Initialize Logger
logger = get_logger("inference")


def main() -> None:
    """Main function to orchestrate the prediction process."""
    parser = argparse.ArgumentParser(description="Run denial prediction on new claims.")
    parser.add_argument(
        "--input",
        default=INFERENCE_INPUT_PATH,
        help="Path to the input CSV file of new claims.",
    )
    parser.add_argument(
        "--output",
        default=DENIAL_PREDICTION_OUTPUT_PATH,
        help="Path to save the output CSV with predictions.",
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
            numeric_transformer=numeric_transformer,
        )

        # Combine original data with predictions
        df_output = pd.concat(
            [df_input.reset_index(drop=True), pd.DataFrame(results)], axis=1
        )

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
