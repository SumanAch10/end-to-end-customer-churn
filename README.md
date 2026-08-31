# End-to-End Customer Churn Prediction

Live-Link: https://end-to-end-customer-churn-1.onrender.com/

A production-style machine learning project that predicts whether a telecom
customer is likely to churn. It covers the full lifecycle: data validation,
preprocessing, feature engineering, model training with MLflow tracking, a
FastAPI prediction service, a Gradio frontend, tests, and Docker images for
both services.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM sample data set, 7,043 customers, 21 columns).

---

## Table of contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Data](#data)
- [Training pipeline](#training-pipeline)
- [Model](#model)
- [Serving](#serving)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Docker](#docker)
- [Google Cloud Run](#google-cloud-run)
- [Testing](#testing)
- [Development notes](#development-notes)

---

## Architecture

Two independent layers. The frontend never touches the model; it only speaks
HTTP to the backend.

```
                        TRAINING (offline)

  data/raw/dataset.csv
          |
          v
  validate_telco_data()        Great Expectations quality gate
          |
          v
  preprocess_data()            trim columns, drop IDs, coerce TotalCharges,
          |                    map Churn -> 0/1, fill numeric NAs
          v
  train / val / test split     stratified, 60 / 20 / 20
          |
          v
  build_features()             ColumnTransformer:
          |                    one-hot (categorical) + passthrough (numeric)
          v
  Pipeline(preprocessor, RandomForestClassifier)
          |
          v
  MLflow run                   params, metrics, fitted pipeline artifact
          |
          v
  artifacts/production_model/  the promoted model used at serving time


                        SERVING (online)

  Browser
     |
     v
  Gradio UI  (:7860)           collects customer fields
     |
     |  HTTP POST /predict
     v
  FastAPI    (:8000)           Pydantic validates the request body
     |
     v
  inference.predict()          builds a one-row DataFrame in training
     |                         column order, applies the threshold
     v
  fitted sklearn Pipeline      ColumnTransformer -> RandomForest
     |
     v
  {prediction, label, churn_probability, threshold}
```

The key design rule: **all preprocessing lives inside the fitted pipeline**.
The serving code does no encoding or scaling of its own, so training and
inference cannot drift apart.

---

## Project structure

```
customer_churn_prediction/
├── data/                          # not versioned (see "Data")
│   ├── raw/dataset.csv
│   └── processed/telco_churn_processed.csv
├── artifacts/
│   └── production_model/          # promoted MLflow model (committed)
├── notebooks/
│   ├── eda.ipynb                  # exploratory data analysis
│   └── inference.ipynb            # loads a run's model, sanity-checks predictions
├── scripts/
│   ├── run_pipeline.py            # full training pipeline + MLflow logging
│   ├── prepare_preprocessed_data.py   # validate + preprocess + save CSV
│   └── deploy_cloudrun.sh         # one-command Cloud Run deploy
├── src/
│   ├── data/
│   │   ├── load_data.py           # CSV -> DataFrame with existence check
│   │   └── preprocess.py          # cleaning shared by training and analysis
│   ├── features/
│   │   ├── build_features.py      # ColumnTransformer factory
│   │   └── pipeline.py            # (work in progress)
│   ├── serving/
│   │   └── inference.py           # loads the model once, predicts
│   ├── app/
│   │   ├── main.py                # FastAPI backend
│   │   ├── cloud_main.py          # FastAPI + Gradio in one Cloud Run service
│   │   └── gradio_ui.py           # Gradio frontend
│   └── utils/
│       ├── utils.py               # logger factory
│       └── validate_telco_data.py # Great Expectations suite
├── tests/
│   ├── test_api.py                # endpoint contract tests
│   ├── test_cloud_main.py         # Cloud Run entrypoint tests
│   └── test_inference.py          # inference-layer tests
├── Dockerfile.api
├── Dockerfile.ui
├── Dockerfile.cloudrun            # combined API + UI image for Cloud Run
├── cloudbuild.yaml                # Cloud Build config (uses Dockerfile.cloudrun)
├── docker-compose.yml
└── requirements.txt
```

---

## Getting started

Requires Python 3.13 (the production model is pickled under 3.13 / scikit-learn
1.9.0; other versions may fail to unpickle).

```bash
git clone https://github.com/SumanAch10/end-to-end-customer-churn.git
cd end-to-end-customer-churn

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the two services in separate terminals:

```bash
# Terminal 1 — backend
uvicorn src.app.main:app --reload --port 8000

# Terminal 2 — frontend
python -m src.app.gradio_ui
```

Then open <http://127.0.0.1:7860> for the UI, or <http://127.0.0.1:8000/docs>
for the interactive API docs.

Both commands must run from the repository root so that `src` resolves as an
import package.

---

## Data

`data/` is intentionally **not** committed. Download the Telco Customer Churn
CSV from Kaggle and place it at:

```
data/raw/dataset.csv
```

To produce the cleaned dataset used by the notebooks:

```bash
python scripts/prepare_preprocessed_data.py
# -> data/processed/telco_churn_processed.csv
```

That script fails loudly if the Great Expectations suite does not pass, so a
malformed CSV never reaches training.

### Validation rules

`src/utils/validate_telco_data.py` enforces:

| Check | Columns |
| --- | --- |
| Value in `{Yes, No}` | `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn` |
| Value in `{Yes, No, No phone service}` | `MultipleLines` |
| Value in `{Yes, No, No internet service}` | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Value in `{DSL, Fiber optic, No}` | `InternetService` |
| Value in `{Month-to-month, One year, Two year}` | `Contract` |
| Value in the four known payment methods | `PaymentMethod` |
| Between 0 and 120 | `tenure` |
| Between 0 and 200 | `MonthlyCharges` |
| Column exists, and is blank or numeric text | `TotalCharges` |

`TotalCharges` gets a regex rule rather than a numeric one because the raw
Kaggle file stores it as text and uses blank strings for 11 new customers.
`preprocess_data` coerces it to numeric and fills those blanks with 0.

---

## Training pipeline

```bash
# MLflow tracking server must be reachable at http://127.0.0.1:5000
mlflow server --host 127.0.0.1 --port 5000

python scripts/run_pipeline.py --target Churn --threshold 0.3 --test_size 0.4
```

Stages, in order:

1. **Load** the raw CSV.
2. **Validate** it with Great Expectations; log `data_quality_pass` and abort on
   failure (failed expectations are logged as a JSON artifact).
3. **Preprocess** and save the cleaned CSV, asserting no rows were dropped.
4. **Split** stratified into 60% train / 20% validation / 20% test.
5. **Build features** from the training frame only, so encoder categories are
   never fitted on validation or test data.
6. **Train** a `Pipeline(preprocessor, RandomForestClassifier)`.
7. **Log** parameters, timing, and metrics for both validation and test:
   precision, recall, F1, ROC-AUC, and average precision (AUPRC).
8. **Log the fitted pipeline** as an MLflow sklearn model.

Hyperparameters are not hard-coded — the script reads them from a previous
tuning run via `MlflowClient.get_run(...)`, so the production training run
reuses exactly the tuned configuration. The run ID is currently pinned inside
`main()`; change it there to retrain from a different tuning run.

### Promoting a model

Copy the fitted pipeline from the chosen MLflow run into
`artifacts/production_model/`. `notebooks/inference.ipynb` shows how to pull a
run's model back with `mlflow.sklearn.load_model("runs:/<run_id>/model")` and
sanity-check it before promotion. Once the directory is in place, serving reads
it directly and never contacts the tracking server.

---

## Model

| Item | Value |
| --- | --- |
| Estimator | `RandomForestClassifier` inside an sklearn `Pipeline` |
| Preprocessing | `OneHotEncoder(handle_unknown="ignore")` on categorical columns, passthrough on numeric |
| Decision threshold | `0.30`, not the default `0.5` |
| Serialization | MLflow sklearn flavour, cloudpickle |
| Tracked metrics | precision, recall, F1, ROC-AUC, AUPRC (validation + test) |

The threshold is deliberately low. Churn is the minority class, and missing a
churner (a lost customer) costs far more than a false alarm (an unnecessary
retention offer), so the model trades precision for recall.

`handle_unknown="ignore"` means an unseen category at inference time becomes an
all-zero block rather than an exception — the API stays up on surprising input.

---

## Serving

### `src/serving/inference.py`

- Loads the fitted pipeline **once** at import time and reuses it for every
  request. A failure to load raises `RuntimeError` immediately at startup
  rather than on the first request.
- Reads the expected raw column list from the fitted preprocessor's
  `feature_names_in_`, so the model itself defines its input contract.
- Reorders the incoming record into training column order before predicting;
  missing features raise `ValueError`.
- Never trains, and never contacts MLflow.

### `src/app/main.py`

FastAPI wrapper. Pydantic `Literal` types reject out-of-domain categories
before they reach the model. `ValueError` from the inference layer maps to
HTTP 422; anything else maps to HTTP 500 with a generic message, while the
full traceback goes to `logs/api.log`.

### `src/app/gradio_ui.py`

A three-column form that posts to the API over HTTP and renders the label and
probability. It imports neither the model nor scikit-learn, so the UI can be
deployed and scaled independently of the backend.

---

## API reference

### `GET /`

Health check.

```json
{ "status": "ok", "service": "telco-churn-api" }
```

### `POST /predict`

Request body — all 19 fields are required:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

Response:

```json
{
  "prediction": 1,
  "label": "Likely to churn",
  "churn_probability": 0.7421,
  "threshold": 0.3
}
```

Status codes: `200` success, `422` validation failure (bad category, negative
number, missing field), `500` inference failure.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @customer.json
```

---

## Configuration

Environment variables, all optional:

| Variable | Default | Used by | Purpose |
| --- | --- | --- | --- |
| `MODEL_PATH` | `artifacts/production_model` | inference | Directory of the MLflow model to serve |
| `CHURN_THRESHOLD` | `0.30` | inference | Probability at or above which a customer is flagged |
| `CHURN_API_URL` | `http://127.0.0.1:8000/predict` | Gradio UI | Backend endpoint |

Logs are written to `logs/api.log` and `logs/inference.log` (not versioned).

---

## Docker

The API and UI ship as separate images so they can scale independently.

```bash
docker compose up --build
```

- API: <http://localhost:8000>
- UI: <http://localhost:7860>

Compose points the UI at `http://api:8000/predict` over the internal network —
`localhost` would resolve to the UI container itself.

Building the images individually:

```bash
docker build -f Dockerfile.api -t telco-churn-api .
docker build -f Dockerfile.ui  -t telco-churn-ui  .

docker run -p 8000:8000 telco-churn-api
docker run -p 7860:7860 -e CHURN_API_URL=http://host.docker.internal:8000/predict telco-churn-ui
```

`Dockerfile.api` bakes `artifacts/production_model` into the image at
`/app/model`, so the container needs no volume mount, no network access, and no
MLflow server to serve predictions.

---

## Google Cloud Run

For Google Cloud, use `Dockerfile.cloudrun`. It serves the Gradio UI and the
prediction API from one Cloud Run service, which is simpler than wiring two
public services together.

Recommended defaults:

- Region: `us-central1`
- Service name: `telco-churn-app`
- Public access: enabled with `--allow-unauthenticated`

### Deploy

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then
authenticate once:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Everything after that is one command. `scripts/deploy_cloudrun.sh` enables the
required APIs, creates the Artifact Registry repository, builds the image and
deploys the service. Each step is idempotent, so re-running it ships an update:

```bash
./scripts/deploy_cloudrun.sh
```

It reads these environment variables, all optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROJECT_ID` | active gcloud project | Target project |
| `REGION` | `us-central1` | Cloud Run and Artifact Registry region |
| `SERVICE` | `telco-churn-app` | Cloud Run service name |
| `REPO` | `telco-churn` | Artifact Registry repository name |
| `MEMORY` | `2Gi` | Memory per instance |
| `CHURN_THRESHOLD` | `0.30` | Classification threshold |

```bash
PROJECT_ID=my-project REGION=europe-west1 ./scripts/deploy_cloudrun.sh
```

<details>
<summary>Equivalent manual commands</summary>

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories create telco-churn \
  --repository-format=docker \
  --location=us-central1

# `gcloud builds submit --tag` only ever builds a file named exactly
# "Dockerfile", so the build goes through cloudbuild.yaml to pick up
# Dockerfile.cloudrun.
gcloud builds submit --config cloudbuild.yaml .

gcloud run deploy telco-churn-app \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/telco-churn/telco-churn-app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars MODEL_PATH=/app/model,CHURN_THRESHOLD=0.30
```

</details>

The script prints the service URL when it finishes. To look it up again later:

```bash
gcloud run services describe telco-churn-app \
  --region us-central1 \
  --format='value(status.url)'
```

Useful endpoints:

- `/` -> Gradio UI
- `/predict` -> POST prediction API
- `/health` -> health check

### Verify the image locally first

The Cloud Run image runs anywhere Docker does, so a failed deploy can be
diagnosed without burning a build:

```bash
docker build -f Dockerfile.cloudrun -t telco-churn-cloudrun:local .
docker run --rm -p 8080:8080 -e PORT=8080 telco-churn-cloudrun:local

curl http://localhost:8080/health
```

Notes:

- The model artifact is baked into the image from `artifacts/production_model`,
  so the container needs no volume mount and no MLflow server.
- The health endpoint is `/health`, **not** the conventional `/healthz`. Cloud
  Run's Google Frontend intercepts the exact path `/healthz` and answers it with
  its own 404, so the request never reaches the container. This is invisible
  locally, where `/healthz` works fine.
- `.gcloudignore` keeps the Cloud Build upload to the few files the image needs.
  Without it, `gcloud` falls back to `.gitignore` and uploads `.git/` too.
- `--min-instances 0` keeps idle cost down, which matters on the $300 free trial.
- `--cpu-boost` gives the instance extra CPU during startup, which shortens the
  cold start caused by importing scikit-learn, MLflow and Gradio.
- `2Gi` memory is a conservative default. A local container serving predictions
  settles around 210 MB, so `1Gi` is likely sufficient if you want to trim it.

---

## Testing

```bash
pytest -q
```

- `tests/test_inference.py` — the inference layer returns a well-formed result:
  binary prediction, probability in `[0, 1]`, threshold echoed back.
- `tests/test_api.py` — health endpoint responds, `/predict` returns a
  prediction, and an invalid category is rejected with 422.

The tests load the real committed model, so they double as a check that the
promoted artifact is loadable in the current environment.

Formatting is enforced with Black:

```bash
black src scripts tests
```

CI (`.github/workflows/ci.yml`) runs the format check and the test suite on
every push and pull request to `main`.

---

## Development notes

Known gaps, kept visible rather than hidden:

- `src/features/pipeline.py` is an unfinished alternative to
  `build_features.py`; nothing imports it.
- `src/utils/validate_telco_data.py` builds its failure list with
  `str[r.expectation_config]`, which raises if any expectation fails. The happy
  path works; the failure path needs fixing before it can report *why*
  validation failed.
- The tuning run ID in `scripts/run_pipeline.py` is hard-coded and should move
  to a CLI argument or config file.
- `build_features.py` executes a module-level `pd.read_csv` on import, so it
  requires `data/raw/dataset.csv` to exist even when only the function is
  needed.
