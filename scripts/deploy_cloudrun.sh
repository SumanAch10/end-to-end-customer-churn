#!/usr/bin/env bash
#
# Build and deploy the combined API + Gradio UI to Google Cloud Run.
#
# Usage:
#   ./scripts/deploy_cloudrun.sh                 # uses the active gcloud project
#   PROJECT_ID=my-project ./scripts/deploy_cloudrun.sh
#   REGION=europe-west1 ./scripts/deploy_cloudrun.sh
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

REGION="${REGION:-us-central1}"
REPO="${REPO:-telco-churn}"
SERVICE="${SERVICE:-telco-churn-app}"
MEMORY="${MEMORY:-2Gi}"
CHURN_THRESHOLD="${CHURN_THRESHOLD:-0.30}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "error: gcloud is not installed. See https://cloud.google.com/sdk/docs/install" >&2
  exit 1
fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "error: no project set. Run 'gcloud config set project YOUR_PROJECT_ID'" >&2
  echo "       or call this script as 'PROJECT_ID=your-project $0'" >&2
  exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"

echo "==> Project: ${PROJECT_ID}"
echo "==> Region:  ${REGION}"
echo "==> Image:   ${IMAGE}"

echo "==> Enabling required APIs (no-op if already enabled)"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "${PROJECT_ID}"

echo "==> Ensuring Artifact Registry repository '${REPO}' exists"
if ! gcloud artifacts repositories describe "${REPO}" \
      --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Telco churn prediction service" \
    --project "${PROJECT_ID}"
else
  echo "    already exists"
fi

echo "==> Building image with Cloud Build"
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions "_REGION=${REGION},_REPO=${REPO},_SERVICE=${SERVICE}" \
  --project "${PROJECT_ID}" \
  .

echo "==> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --port 8080 \
  --memory "${MEMORY}" \
  --cpu 1 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "MODEL_PATH=/app/model,CHURN_THRESHOLD=${CHURN_THRESHOLD}" \
  --project "${PROJECT_ID}"

URL="$(gcloud run services describe "${SERVICE}" \
        --region "${REGION}" \
        --project "${PROJECT_ID}" \
        --format='value(status.url)')"

echo
echo "==> Deployed: ${URL}"
echo "    UI      : ${URL}/"
echo "    API     : ${URL}/predict"
echo "    Health  : ${URL}/health"

echo
echo "==> Verifying health endpoint"
curl -fsS "${URL}/health" && echo
