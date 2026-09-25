"""Modal deployment for the existing FastAPI app and analysis pipeline."""

from __future__ import annotations

import modal

from appstore_reviews.keywords import DEFAULT_KEYWORD_MODEL, DEFAULT_KEYWORD_REVISION
from appstore_reviews.sentiment import DEFAULT_SENTIMENT_MODEL, DEFAULT_SENTIMENT_REVISION


HF_HOME = "/opt/huggingface"
FASTTEXT_MODEL_PATH = "/opt/models/lid.176.ftz"
DATA_MOUNT = "/data"


def _download_models(
    sentiment_model: str,
    sentiment_revision: str,
    keyword_model: str,
    keyword_revision: str,
    fasttext_model_path: str,
) -> None:
    """Populate the image's model cache at build time, never at cold start."""
    from pathlib import Path
    from urllib.request import urlretrieve

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=sentiment_model, revision=sentiment_revision)
    snapshot_download(repo_id=keyword_model, revision=keyword_revision)
    model_path = Path(fasttext_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        model_path,
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["sentiment", "keywords"])
    .env({
        "HF_HOME": HF_HOME,
        "FASTTEXT_MODEL_PATH": FASTTEXT_MODEL_PATH,
        "APPSTORE_SCAN_DIR": f"{DATA_MOUNT}/scans",
    })
    # The build function imports this module, which imports appstore_reviews.
    # Copy the package before run_function so it exists during image builds.
    .add_local_python_source("appstore_reviews", copy=True)
    .run_function(
        _download_models,
        args=(
            DEFAULT_SENTIMENT_MODEL,
            DEFAULT_SENTIMENT_REVISION,
            DEFAULT_KEYWORD_MODEL,
            DEFAULT_KEYWORD_REVISION,
            FASTTEXT_MODEL_PATH,
        ),
        timeout=3600,
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
)

app = modal.App("appstore-reviews-analysis")
volume = modal.Volume.from_name("appstore-reviews-data", create_if_missing=True)
secrets = [modal.Secret.from_name("appstore-reviews-secrets")]


@app.function(
    image=image,
    secrets=secrets,
    volumes={DATA_MOUNT: volume},
    cpu=2,
    memory=4096,
    timeout=3600,
    max_containers=1,
)
def analyze_worker(scan_id: str, payload: dict) -> None:
    from appstore_reviews import scan_runtime
    from appstore_reviews.api import AnalyzeRequest, _run_analysis_job

    scan_runtime.configure(reload_volume=volume.reload, commit_volume=volume.commit)
    _run_analysis_job(scan_id, AnalyzeRequest.model_validate(payload))


@app.function(
    image=image,
    secrets=secrets,
    volumes={DATA_MOUNT: volume},
    cpu=1,
    memory=2048,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    from appstore_reviews import scan_runtime
    from appstore_reviews.api import app as web_app

    scan_runtime.configure(
        reload_volume=volume.reload,
        commit_volume=volume.commit,
        dispatch=lambda scan_id, payload: analyze_worker.spawn(scan_id, payload),
    )
    return web_app
