# =============================================================================
# Configuration
# =============================================================================
# Define file paths as variables. This is the single source of truth.
TRAINING_DATA_RAW = data/raw_claims_v2.csv
PROCESSED_DATA = data/processed_claims.csv
INFERENCE_DATA = data/inference_claims.csv

MODEL_DIR = models
PREDICTION_MODEL = $(MODEL_DIR)/denial_prediction_model.joblib
CLUSTER_MODEL = $(MODEL_DIR)/cluster_model.joblib
REDUCER_MODEL = $(MODEL_DIR)/reducer_model.joblib

PREDICTION_OUTPUT = data/denial_prediction_output.csv
CLUSTERING_OUTPUT = data/clustering_output.csv
FINAL_ROUTING_FILE = data/routing_ready_claims.csv
CLUSTER_LABELS_JSON = data/cluster_labels.json

# =============================================================================
# Main Workflows (User-Facing Commands)
# =============================================================================

# Default command when you just type "make"
all: route-new-claims

# To run the full end-to-end cycle for a demo
demo: clean generate-data retrain-all route-new-claims
	@echo "====================================================================="
	@echo "DEMO PIPELINE COMPLETE!"
	@echo "Final triaged claims are in: $(FINAL_ROUTING_FILE)"
	@echo "====================================================================="

# To run the periodic analytics and model training pipeline
retrain-all: $(PREDICTION_MODEL) $(CLUSTER_MODEL) $(REDUCER_MODEL)
	@echo "--> All models have been successfully retrained."

# To run the daily operational pipeline on new claims
route-new-claims: $(FINAL_ROUTING_FILE)
	@echo "--> Inference pipeline complete. Final output is at $(FINAL_ROUTING_FILE)"



# Command to launch the Streamlit UI
ui:
	@echo "--> Launching the Streamlit dashboard at http://localhost:8501"
	streamlit run streamlit_app.py

run-api:
	@echo "--> Starting FastAPI backend server at http://localhost:8000"
	uvicorn app.main:app --host 0.0.0.0 --port 8000


# =============================================================================
# Code Quality & Hooks
# =============================================================================
lint:
	black . && ruff check .

test:
	pytest

check: lint test

precommit:
	pre-commit install


# =============================================================================
# Pipeline Dependencies (The "Magic" of Make)
# =============================================================================

# --- Inference Pipeline Dependencies ---
$(FINAL_ROUTING_FILE): scripts/combine_predictions_and_clusters.py $(PREDICTION_OUTPUT) $(CLUSTERING_OUTPUT)
	@echo "--> (3/3) Merging predictions and clusters..."
	python scripts/combine_predictions_and_clusters.py --predictions $(PREDICTION_OUTPUT) --clusters $(CLUSTERING_OUTPUT) --output $@

$(CLUSTERING_OUTPUT): scripts/cluster_denials.py $(PREDICTION_OUTPUT) $(CLUSTER_MODEL) $(REDUCER_MODEL)
	@echo "--> (2/3) Assigning clusters to predicted denials..."
	python scripts/cluster_denials.py \
		--input $(PREDICTION_OUTPUT) \
		--output $@ \
		--reducer-model $(REDUCER_MODEL) \
		--cluster-model $(CLUSTER_MODEL)

$(PREDICTION_OUTPUT): scripts/predict_denials.py $(INFERENCE_DATA) $(PREDICTION_MODEL)
	@echo "--> (1/3) Running denial prediction on new claims..."
	python scripts/predict_denials.py --input $(INFERENCE_DATA) --output $@


# --- Training Pipeline Dependencies ---
$(PREDICTION_MODEL): claimtriageai/prediction/train_denial_model.py $(PROCESSED_DATA)
	@echo "--> Training denial prediction model..."
	python -m claimtriageai.prediction.train_denial_model --data $(PROCESSED_DATA)

# CORRECTED RULE: This now uses the raw training data as input
$(CLUSTER_MODEL) $(REDUCER_MODEL): claimtriageai/clustering/root_cause_cluster.py $(TRAINING_DATA_RAW)
	@echo "--> Training clustering model and UMAP reducer..."
	python -m claimtriageai.clustering.root_cause_cluster --input $(TRAINING_DATA_RAW)

$(PROCESSED_DATA): claimtriageai/preprocessing/build_features.py $(TRAINING_DATA_RAW)
	@echo "--> Building features from training data..."
	python -m claimtriageai.preprocessing.build_features --input $(TRAINING_DATA_RAW) --output $@


# =============================================================================
# Utility Commands
# =============================================================================
generate-data:
	python scripts/mock_data_creation.py
	python scripts/mock_inference_data_creation.py

# Plot umap of clusters
plot-umap: $(FINAL_ROUTING_FILE) $(CLUSTER_LABELS_JSON)
	@echo "--> Plotting UMAP of clusters from $(FINAL_ROUTING_FILE)"
	python scripts/plot_umap_clusters.py \
		--input $(FINAL_ROUTING_FILE) \
		--labels $(CLUSTER_LABELS_JSON)

clean:
	rm -f data/*.csv
	rm -f $(MODEL_DIR)/*.joblib


# Tell Make which targets are not files
.PHONY: all demo lint test check precommit retrain-all generate-data clean route-new-claims ui