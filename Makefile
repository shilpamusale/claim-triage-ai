lint:
	black . && ruff check .

typecheck:
	mypy .

test:
	pytest

check: lint typecheck test

precommit:
	pre-commit install

prepush:
	chmod +x pre_push_checks.sh && ./pre_push_checks.sh

# Step 1: Preprocessing/Feature Engineering
build-features:
	python -m claimtriageai.preprocessing.build_features --output $(INFERENCE_INPUT_PATH)

# Step 2: Model Training (optional)
train-denial-model:
	python -m claimtriageai.prediction.train_denial_model

# Step 3: Prediction
predict-denials:
	python scripts/predict_denials.py --input $(INFERENCE_INPUT_PATH) --output $(DENIAL_PREDICTION_OUTPUT_PATH)

# Step 4: Clustering (optional)
cluster-denials:
	python scripts/cluster_denials.py --input $(DENIAL_PREDICTION_OUTPUT_PATH) --output $(CLUSTERING_OUTPUT_PATH)

# Step 5: Visualization (optional)
plot-umap-clusters:
	python scripts/plot_umap_clusters.py --input $(CLUSTERING_OUTPUT_PATH)

# Step 6: Combine Results (optional)
combine-predictions-clusters:
	python scripts/combine_predictions_and_clusters.py --predictions $(DENIAL_PREDICTION_OUTPUT_PATH) --clusters $(CLUSTERING_OUTPUT_PATH) --output $(MERGED_ROUTING_PATH)

# Step 7: Routing Policy
run-routing-policy:
	python scripts/run_routing_policy.py --input $(MERGED_ROUTING_PATH) --output $(CLAIMS_ROUTING_OUTPUT_PATH)

# Step 8: API Serving (optional)
serve-api:
	uvicorn app.main:app --reload

# Step 9: Streamlit App (optional)
streamlit:
	streamlit run streamlit_app.py

# Path variables from claimtriageai/configs/paths.py
INFERENCE_INPUT_PATH=data/inference_input.csv
DENIAL_PREDICTION_OUTPUT_PATH=data/denial_prediction_output.csv
CLUSTERING_OUTPUT_PATH=data/clustering_output.csv
MERGED_ROUTING_PATH=data/merged_denial-prediction_clustering.csv
CLAIMS_ROUTING_OUTPUT_PATH=data/routed_claims.csv

# Run full pipeline
run-pipeline:
	make build-features
	make train-denial-model
	make predict-denials
	make cluster-denials
	make plot-umap-clusters
	make combine-predictions-clusters
	make run-routing-policy