"""Fetch fine-tuned Keras weights from a GitHub Release if missing under streamlit_app/models/."""

import os
import urllib.error
import urllib.request

OWNER, REPO, TAG = "arbutler2003", "Pneuomonia-Detection-CNN", "v1.0.0"
MODEL_NAME = "best_cropped_finetuned.keras"
MODEL_URL = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{MODEL_NAME}"
_USER_AGENT = "pneumonia-detection-streamlit"
_CHUNK = 4 * 1024 * 1024


def _open(url: str, timeout: int):
    headers = {"User-Agent": _USER_AGENT}
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout
    )


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
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Could not download model from release asset URL: HTTP {exc.code}"
        ) from exc
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
    _download(MODEL_URL, path)
    return path
