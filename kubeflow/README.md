# Kubeflow Pipelines

This directory contains the Kubeflow Pipelines v2 orchestration layer for the
startup acquisition-status project. Pipeline tasks use one reusable image for
data preparation, training, evaluation, and packaging. The Streamlit app keeps
its separate root `Dockerfile` and is not started inside pipeline task pods.

## Folder structure

```text
docker/
└── Dockerfile.pipeline                 # Reusable non-root task image

requirements-kubeflow.txt               # Compiler/submission SDK only

src/churn_model/kubeflow/
├── __init__.py
├── runtime.py                           # Container command adapters
├── components.py                        # KFP container-component definitions
├── pipeline.py                          # Training DAG
└── compile_pipeline.py                  # IR YAML compiler command

kubeflow/
├── README.md
└── compiled/
    ├── .gitkeep
    └── startup_training_pipeline.yaml   # Generated; excluded from image context
```

The task image contains the project source and the packages in
`requirements.txt`. It deliberately does not install the KFP SDK. KFP is only
needed on the workstation or notebook that compiles and submits the pipeline,
so install `requirements-kubeflow.txt` there.

## 1. Install the compiler SDK

From the repository root:

```powershell
python -m pip install -r requirements-kubeflow.txt
```

The SDK and Kubernetes plugin are bounded to compatible KFP v2 release lines.
They should also be compatible with the KFP v2 backend installed in the target
cluster. Check the cluster version before changing those bounds.

## 2. Build and push the task image

Every Kubernetes node that may run a task must be able to pull the image. Use an
immutable tag such as the Git commit rather than `latest`.

```powershell
$Registry = "REGISTRY/USER"
$Tag = git rev-parse --short=12 HEAD
$Image = "${Registry}/global-tech-startup-pipeline:${Tag}"

docker buildx build `
    --platform linux/amd64 `
    --file docker/Dockerfile.pipeline `
    --tag $Image `
    --push `
    .
```

Replace `REGISTRY/USER` with a registry path reachable from the cluster, for
example `docker.io/my-user` or a private organization registry. Use another
platform only when the Kubernetes nodes and all Python wheels support it.

For a local build that is not pushed:

```powershell
docker build --file docker/Dockerfile.pipeline --tag global-tech-startup-pipeline:dev .
docker run --rm global-tech-startup-pipeline:dev `
    python -m src.churn_model.kubeflow.runtime --help
```

The image runs as UID and GID `10001`. It contains no raw dataset or trained
model and has no Streamlit server command. Component definitions replace the
diagnostic default command when KFP starts each task.

## 3. Compile the pipeline

Compilation records the component image in the generated IR YAML for the
`startup-acquisition-training` pipeline. It does not run training and does not
upload a local dataset.

```powershell
python -m src.churn_model.kubeflow.compile_pipeline `
    --component-image $Image `
    --output kubeflow/compiled/startup_training_pipeline.yaml
```

If the compiler reports that the output already exists, add `--overwrite` only
when replacing that generated IR is intentional:

```powershell
python -m src.churn_model.kubeflow.compile_pipeline `
    --component-image $Image `
    --output kubeflow/compiled/startup_training_pipeline.yaml `
    --overwrite
```

For a reproducible release, prefer an image digest such as
`REGISTRY/USER/global-tech-startup-pipeline@sha256:...` as the
`--component-image` value. A digest prevents a tag from resolving to different
code in a later run.

## 4. Configure artifact persistence

KFP tasks run in different pods. They must exchange datasets, models, manifests,
and metrics through KFP artifacts rather than local paths such as `data/`,
`models/`, or `C:\Users\...`.

The pipeline uses typed artifacts for large outputs:

- `Dataset` for source, cleaned, split, and prediction tables.
- `Model` for the trusted trained classifier.
- `Artifact` for the matching manifest and other structured files.
- `Metrics` for values displayed and tracked by KFP.

Each component writes to the KFP-provided artifact path. The KFP launcher copies
that output to its artifact URI under the pipeline root. Input artifact paths
must be treated as read-only.

Before running, configure a durable pipeline root supported by the cluster, such
as SeaweedFS, S3, or GCS. The pipeline service account needs read/write access to
that location. Prefer the administrator-managed cluster default. If the cluster
has no suitable default, set `pipeline_root` while creating the run in the UI or
SDK. Do not use the task container filesystem as persistent storage.

A PVC is only needed for unusually large shared scratch data or a legacy tool
that requires shared filesystem semantics. When a PVC is used, its storage class
and permissions must support UID/GID `10001` (commonly through an appropriate
`fsGroup` policy), and tasks sharing it need an explicit dependency order.

## 5. Private registry access

For a private registry, create an image-pull Secret in the same Kubeflow profile
namespace as the pipeline run:

```powershell
$Namespace = "YOUR_PROFILE_NAMESPACE"

kubectl --namespace $Namespace create secret docker-registry pipeline-registry `
    --docker-server="REGISTRY_HOST" `
    --docker-username="REGISTRY_USER" `
    --docker-password="$env:REGISTRY_TOKEN"
```

Then expose `pipeline-registry` to task pods in one of these ways:

1. Configure the component tasks with
   `kfp.kubernetes.set_image_pull_secrets(task, ["pipeline-registry"])`.
2. Attach the Secret to the Kubernetes ServiceAccount used by pipeline task
   pods, following the policy of the Kubeflow installation.

Secrets are namespaced. A Secret created in another namespace cannot be used by
the run. Do not place registry credentials in the repository or compiled YAML.
Use `IfNotPresent` with immutable tags or digests so nodes can reuse a verified
cached image.

## 6. Upload and run

The simplest first run is through the Kubeflow Pipelines UI:

1. Open **Pipelines** and upload
   `kubeflow/compiled/startup_training_pipeline.yaml`.
2. Open **Runs**, create a run from the uploaded pipeline, and select the target
   experiment.
3. Supply `source_dataset_uri` and, when changing the gate, set
   `minimum_approval_metric`. These are run parameters; the source is not a local
   filesystem path and neither value needs to be embedded at compile time.
4. Confirm or override the pipeline root, then start the run.

The compiled package can also be uploaded with the KFP CLI after its endpoint
and authentication have been configured:

```powershell
kfp pipeline create `
    --pipeline-name "startup-acquisition-training" `
    kubeflow/compiled/startup_training_pipeline.yaml
```

For a scripted run with required parameters, use the SDK client:

```python
from kfp import Client

client = Client(host="https://YOUR_KUBEFLOW_ENDPOINT/pipeline")
run = client.create_run_from_pipeline_package(
    "kubeflow/compiled/startup_training_pipeline.yaml",
    experiment_name="global-tech-startup",
    arguments={
        "source_dataset_uri": "s3://YOUR_BUCKET/startups/global_tech_startups_2026.csv",
        "minimum_approval_metric": 0.70,
    },
    pipeline_root="s3://YOUR_BUCKET/kfp-artifacts",
)
print(run.run_id)
```

Replace the example S3 locations with a URI scheme supported and configured in
the target KFP installation. If the administrator has configured a suitable
default pipeline root, omit the `pipeline_root` argument.

For a multi-user Kubeflow installation, submit into the intended profile
namespace and use that installation's supported authentication method. An
in-cluster Kubeflow Notebook commonly uses its projected ServiceAccount token;
an external workstation usually needs the Kubeflow endpoint and identity-provider
authentication. The compiled YAML itself contains no credentials.

## Current startup defaults

The initial Kubeflow pipeline follows the existing project training defaults:

| Setting | Default |
| --- | --- |
| Dataset format | CSV |
| Target column | `acquisition_status` |
| Dropped identifier | `company_id` |
| Train / validation / test | `0.70 / 0.15 / 0.15` |
| Random state | `42` |
| Model type | `logistic_regression` |
| Maximum iterations | `2000` |
| Random-forest estimators | `300` when that model is selected |
| Class weight | `balanced` |
| Founding-year reference | `2026` |
| Minimum approval metric | `0.70` |
| Packaged model filename | `startup_classifier.pkl` |

The project also derives company age, logarithmic funding/valuation/revenue
features, and the configured funding-per-employee and valuation-to-revenue
ratios. Change a run parameter only when the resulting model and manifest remain
paired; inference must consume artifacts from the same training run.

`source_dataset_uri` and `minimum_approval_metric` are supplied when the run is
created. The separate `approval_metric` parameter selects which evaluation
metric is compared with the threshold. A failed quality gate is an intentional
pipeline failure: inspect the evaluation metrics instead of promoting that
model.

## Operational notes

- Pipeline tasks log to standard output so their messages are visible in the
  KFP UI. The image also leaves `/app/logs` writable for compatibility with the
  project's rotating file logger; that directory is not durable.
- Never load a pickle supplied by an untrusted user. Inference should consume
  only the model artifact produced by this trusted training pipeline.
- Recompile after changing component code, arguments, the image digest, or the
  DAG. A new source dataset or quality threshold normally requires only a new
  run, not recompilation.
- Use KFP's artifact lineage and run metadata to identify exactly which dataset,
  image, model, manifest, and metrics belong together.
