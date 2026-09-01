# Cloud Run deployment

One-time setup (Aamir, since he owns the GCP project) plus what Ana needs
to configure in GitHub so CI publishes the image automatically.

## 1. One-time GCP setup (Aamir)

Pick a region once and reuse it everywhere below — e.g. `europe-west1`.
Everything (Artifact Registry repo, Cloud Run service) must live in the
same region.

```bash
gcloud config set project <GCP_PROJECT_ID>

# Enable the two APIs this needs.
gcloud services enable artifactregistry.googleapis.com run.googleapis.com

# Create the Artifact Registry repository CI will push images into.
gcloud artifacts repositories create mental-health-repo \
  --repository-format=docker \
  --location=<GCP_REGION> \
  --description="Mental Health Intelligence — FastAPI service images"

# Service account CI will authenticate as, scoped to just push images —
# not a broad project-owner credential.
gcloud iam service-accounts create github-actions-deployer \
  --display-name="GitHub Actions — Artifact Registry pusher"

gcloud artifacts repositories add-iam-policy-binding mental-health-repo \
  --location=<GCP_REGION> \
  --member="serviceAccount:github-actions-deployer@<GCP_PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Key CI will use to authenticate — the JSON this prints is the ONLY
# secret in this whole setup. Send it to Ana over a private channel
# (not this group chat / not committed anywhere), she pastes it whole as
# a GitHub secret (step 2 below), then delete the local copy.
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-deployer@<GCP_PROJECT_ID>.iam.gserviceaccount.com
```

## 2. GitHub repo configuration (Ana)

Settings → Secrets and variables → Actions:

**Variables** tab (not secret, plain config):
| Name | Value |
|---|---|
| `GCP_PROJECT_ID` | Aamir's GCP project ID |
| `GCP_REGION` | the region chosen above, e.g. `europe-west1` |
| `GCP_AR_REPO` | `mental-health-repo` |

**Secrets** tab:
| Name | Value |
|---|---|
| `GCP_SA_KEY` | the whole contents of `github-actions-key.json` Aamir sent |

Once these four are set, every push to `main` builds the image and
publishes it to
`<GCP_REGION>-docker.pkg.dev/<GCP_PROJECT_ID>/<GCP_AR_REPO>/mental-health-api:latest`
— see `.github/workflows/ci.yml`.

## 3. Deploying the service (Aamir, after the first image is published)

```bash
gcloud run deploy mental-health-api \
  --image=<GCP_REGION>-docker.pkg.dev/<GCP_PROJECT_ID>/mental-health-repo/mental-health-api:latest \
  --region=<GCP_REGION> \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --memory=2Gi \
  --cpu=2 \
  --set-env-vars="MLFLOW_TRACKING_URI=postgresql://aamir_cloudrun:<PGPASSWORD>@ep-fancy-dream-ayjyumxu-pooler.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require" \
  --set-env-vars="MLFLOW_ARTIFACT_ROOT=s3://mental-health-mlops/mlflow-artifacts" \
  --set-env-vars="AWS_ENDPOINT_URL_S3=https://br-odd-rice-ayn2901i.storage.c-5.us-east-2.aws.neon.tech" \
  --set-env-vars="AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>" \
  --set-env-vars="AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>" \
  --set-env-vars="AWS_REGION=us-east-2"
```

Fill in `<PGPASSWORD>`, `<AWS_ACCESS_KEY_ID>`, `<AWS_SECRET_ACCESS_KEY>`
from the credential files Ana sent him directly (never commit these, never
paste them in a group chat/issue). `AWS_REGION` and `AWS_ENDPOINT_URL_S3`
must match the S3-compatible storage's actual region/endpoint exactly —
double-check against the credential file rather than retyping from memory.

`--memory=2Gi --cpu=2` (Cloud Run's default is 512Mi/1 CPU) — needed
because the Docker image now installs the `transformers` extra so the API
can serve a DistilBERT candidate if `promote.py` ever puts one in
"production" (see the Dockerfile's comment): torch plus a loaded
DistilBERT checkpoint does not fit in the default 512Mi and would crash
the container with an OOM at startup. If production is a lightweight
scikit-learn model (the common case), this is more memory than strictly
needed, but Cloud Run only bills for actual CPU/memory *used* during
request handling (not a fixed reservation cost the way a VM would be), so
the safety margin is cheap relative to an OOM crash on the day a
DistilBERT version does get promoted.

**Better than `--set-env-vars` for the two actual secrets** (`MLFLOW_TRACKING_URI`
embeds the DB password; `AWS_SECRET_ACCESS_KEY` is the S3 secret): once
this is running reliably, move those two into
[Secret Manager](https://cloud.google.com/secret-manager) and reference
them with `--set-secrets` instead — env vars set this way are visible to
anyone with read access to the Cloud Run service's revision config, not
just people who should see the credentials.

## 4. Redeploying after a new champion is promoted

The model is loaded once at container startup (FastAPI lifespan, see
`api/model_loader.py`) — it is **never** reloaded automatically. After
every weekly retrain + promotion, the service needs an explicit restart to
pick up the new "production" model:

```bash
gcloud run services update mental-health-api --region=<GCP_REGION>
```

(`update` with no changed flags still forces a new revision, which reloads
the container — this is what actually gets the new model in front of
traffic. A regular redeploy from `main` — a new git push — also works,
since it publishes a fresh image and revision either way.)
