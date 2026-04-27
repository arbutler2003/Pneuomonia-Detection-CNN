"""Fetch fine-tuned Keras weights from a GitHub Release if missing under streamlit_app/models/."""

import json
import os
import urllib.error
import urllib.request

OWNER, REPO, TAG = "arbutler2003", "Pneuomonia-Detection-CNN", "v1.0.0"
MODEL_NAME = "best_cropped_finetuned.keras"
_USER_AGENT = "pneumonia-detection-streamlit"
_CHUNK = 4 * 1024 * 1024


def _open(url: str, timeout: int, *, github_api: bool = False):
    headers = {"User-Agent": _USER_AGENT}
    if github_api:
        headers["Accept"] = "application/vnd.github+json"
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout
    )


def _release_keras_download_url() -> str:
    api = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}"
    try:
        with _open(api, 60, github_api=True) as resp:
            assets = json.load(resp).get("assets", [])
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Could not read GitHub release {TAG}: HTTP {exc.code}") from exc

    pairs = [
        (n, u)
        for a in assets
        for n, u in [(a.get("name") or "", a.get("browser_download_url"))]
        if n.endswith(".keras") and u
    ]
    if not pairs:
        raise FileNotFoundError(f"No .keras asset on release {TAG}.")

    for name, url in pairs:
        if name == MODEL_NAME:
            return url
    return pairs[0][1]


def _download(url: str, dest: str) -> None:
    tmp = dest + ".partial"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        with _open(url, 600) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def ensure_best_model_path(streamlit_app_dir: str) -> str:
    """Return path to models/best_cropped_finetuned.keras, downloading from the release if absent."""
    models_dir = os.path.join(streamlit_app_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, MODEL_NAME)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    _download(_release_keras_download_url(), path)
    return path
