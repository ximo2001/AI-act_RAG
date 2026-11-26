"""Download the models used by the project into ./models.

Default source is Hugging Face (what you want on a laptop):

    uv run python scripts/download_models.py

Both chat models are downloaded by default: Qwen2.5-3B-Instruct Q4_K_M (the
default LLM, ~1.9 GB) and Qwen2.5-1.5B-Instruct Q4_K_M (the fast option, ~1 GB).
``--llm 3b`` or ``--llm 1.5b`` downloads only one; ``--skip-llm`` none.

``--source dockerhub`` is the fallback for networks where huggingface.co is
blocked: GGUF files from Docker Hub's
``ai/*`` model artifacts (the 3B is a ready Q4_K_M file; the 1.5B is the F16
file, quantised to Q4_K_M locally) and sentence-transformers folders from
Weaviate's inference images (which bake the original HF weights). Plain
transformers folders get a sentence-transformers pooling config written next
to them. GGUF files already present in ./models are not downloaded again.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import httpx

MODELS_DIR = Path("models")

# name -> (HF repo, Docker Hub repo:tag, pooling) for sentence-transformers models.
ST_MODELS = {
    "bge-small-en-v1.5": (
        "BAAI/bge-small-en-v1.5",
        "semitechnologies/transformers-inference:baai-bge-small-en-v1.5",
        "cls",
    ),
    "paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "semitechnologies/transformers-inference:"
        "sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2",
        "mean",
    ),
    "ms-marco-MiniLM-L-6-v2": (
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "semitechnologies/reranker-transformers:cross-encoder-ms-marco-MiniLM-L-6-v2",
        "",  # cross-encoder: no pooling config
    ),
}


@dataclass(frozen=True)
class GGUFModel:
    """Where to get one Q4_K_M chat model.

    Attributes:
        name: File name under models/ (what ``AIACT_LLM_MODEL`` points to).
        hf_repo: Hugging Face repository publishing ``name``.
        docker_ref: Docker Hub ``ai/*`` artifact (``repo:tag``) with a GGUF layer.
        docker_is_f16: True if that artifact is F16 and must be quantised locally.
    """

    name: str
    hf_repo: str
    docker_ref: str
    docker_is_f16: bool = False


# Chat models, keyed by the ``--llm`` choice. The 3B is the default LLM (DECISIONS D12).
GGUF_MODELS = {
    "3b": GGUFModel(
        name="qwen2.5-3b-instruct-q4_k_m.gguf",
        hf_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        docker_ref="ai/qwen2.5:3B-Q4_K_M",
    ),
    "1.5b": GGUFModel(
        name="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        docker_ref="ai/qwen2.5:1.5B-F16",
        docker_is_f16=True,
    ),
}

_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ]
)


def _token(repo: str) -> str:
    """Get an anonymous pull token for a Docker Hub repository.

    Args:
        repo: e.g. ``"ai/qwen2.5"``.

    Returns:
        Bearer token.
    """
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    return str(httpx.get(url, timeout=30).json()["token"])


def _layers(ref: str) -> tuple[str, str, list[dict[str, object]]]:
    """Resolve an image/artifact reference to its amd64 layers.

    Args:
        ref: ``repo:tag``.

    Returns:
        (repo, token, layers).
    """
    repo, tag = ref.split(":", 1)
    tok = _token(repo)
    headers = {"Authorization": f"Bearer {tok}", "Accept": _ACCEPT}
    base = f"https://registry-1.docker.io/v2/{repo}/manifests"
    manifest = httpx.get(f"{base}/{tag}", headers=headers, timeout=30).raise_for_status().json()
    if "manifests" in manifest:  # multi-arch index -> pick linux/amd64
        digest = next(
            m["digest"]
            for m in manifest["manifests"]
            if m.get("platform", {}).get("architecture") in ("amd64", None)
        )
        manifest = (
            httpx.get(f"{base}/{digest}", headers=headers, timeout=30).raise_for_status().json()
        )
    return repo, tok, manifest["layers"]


def _download_blob(repo: str, tok: str, digest: str, dest: Path) -> None:
    """Stream a blob to disk.

    Args:
        repo: Repository.
        tok: Bearer token.
        digest: Blob digest.
        dest: Output file.
    """
    url = f"https://registry-1.docker.io/v2/{repo}/blobs/{digest}"
    with httpx.stream(
        "GET", url, headers={"Authorization": f"Bearer {tok}"}, follow_redirects=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for part in resp.iter_bytes(1 << 20):
                f.write(part)


def write_sentence_transformers_config(model_dir: Path, pooling: str) -> None:
    """Add modules.json + pooling config so sentence-transformers pools correctly.

    A plain transformers folder would otherwise default to mean pooling, which
    is wrong for BGE (trained with the [CLS] token).

    Args:
        model_dir: Folder with config.json / model.safetensors.
        pooling: ``"cls"`` or ``"mean"``.
    """
    dim = json.loads((model_dir / "config.json").read_text())["hidden_size"]
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
        {
            "idx": 2,
            "name": "2",
            "path": "2_Normalize",
            "type": "sentence_transformers.models.Normalize",
        },
    ]
    (model_dir / "modules.json").write_text(json.dumps(modules, indent=2))
    (model_dir / "1_Pooling").mkdir(exist_ok=True)
    (model_dir / "1_Pooling" / "config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": dim,
                "pooling_mode_cls_token": pooling == "cls",
                "pooling_mode_mean_tokens": pooling == "mean",
            },
            indent=2,
        )
    )
    (model_dir / "2_Normalize").mkdir(exist_ok=True)


def st_from_dockerhub(name: str, ref: str, pooling: str) -> None:
    """Extract ``/app/models/model`` from a Weaviate inference image.

    Args:
        name: Target folder name under models/.
        ref: Image reference.
        pooling: Pooling for the sentence-transformers config ("" for cross-encoders).
    """
    repo, tok, layers = _layers(ref)
    target = MODELS_DIR / name
    # The model lives in a small layer near the end; skip the multi-GB pip layer.
    for layer in reversed(layers):
        size = int(str(layer["size"]))
        if size > 2_000_000_000 or size < 1_000_000:
            continue
        tmp = MODELS_DIR / f".{name}.tgz"
        _download_blob(repo, tok, str(layer["digest"]), tmp)
        with tarfile.open(tmp, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.startswith("app/models/model/")]
            if not members:
                tmp.unlink()
                continue
            target.mkdir(parents=True, exist_ok=True)
            for m in members:
                if not m.isfile():
                    continue
                data = tar.extractfile(m)
                assert data is not None
                out = target / Path(m.name).relative_to("app/models/model")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data.read())
        tmp.unlink()
        if pooling:
            write_sentence_transformers_config(target, pooling)
        print(f"{name}: extracted to {target}")
        return
    raise RuntimeError(f"No model layer found in {ref}")


def quantize_gguf(src: Path, dest: Path) -> None:
    """Quantise an F16 GGUF to Q4_K_M with llama.cpp's built-in quantiser.

    Args:
        src: F16 GGUF.
        dest: Output GGUF.
    """
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
    rc = llama_cpp.llama_model_quantize(str(src).encode(), str(dest).encode(), params)
    if rc != 0:
        raise RuntimeError(f"Quantisation failed with code {rc}")


def gguf_from_dockerhub(model: GGUFModel) -> None:
    """Download a GGUF artifact from Docker Hub (quantising it if it is F16).

    Args:
        model: Chat model to fetch.
    """
    dest = MODELS_DIR / model.name
    if dest.exists():
        print(f"LLM: {dest} (already present)")
        return
    repo, tok, layers = _layers(model.docker_ref)
    layer = next(x for x in layers if "gguf" in str(x.get("mediaType")))
    if not model.docker_is_f16:
        tmp = dest.with_suffix(".part")
        _download_blob(repo, tok, str(layer["digest"]), tmp)
        tmp.rename(dest)
    else:
        f16 = MODELS_DIR / model.name.replace("q4_k_m", "f16")
        if not f16.exists():
            tmp = f16.with_suffix(".part")
            _download_blob(repo, tok, str(layer["digest"]), tmp)
            tmp.rename(f16)
        # Quantise to a temporary name: a failed run must not leave a partial file
        # that the next run would take as "already present".
        tmp = dest.with_suffix(".part")
        quantize_gguf(f16, tmp)
        tmp.rename(dest)
    print(f"LLM: {dest}")


def from_huggingface(llms: list[GGUFModel]) -> None:
    """Download everything from Hugging Face (needs ``huggingface_hub``, part of the ml extra).

    Args:
        llms: Chat models to download (empty: only the embedding/reranker models).
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    for name, (hf_repo, _, _) in ST_MODELS.items():
        snapshot_download(hf_repo, local_dir=MODELS_DIR / name)
        print(f"{name}: {MODELS_DIR / name}")
    for model in llms:
        path = hf_hub_download(model.hf_repo, model.name, local_dir=MODELS_DIR)
        print(f"LLM: {path}")


def main() -> int:
    """Parse arguments and download.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["huggingface", "dockerhub"], default="huggingface")
    parser.add_argument(
        "--llm",
        nargs="+",
        choices=sorted(GGUF_MODELS),
        default=list(GGUF_MODELS),
        help="chat models to download (default: both)",
    )
    parser.add_argument("--skip-llm", action="store_true", help="only embedding/reranker models")
    args = parser.parse_args()
    llms = [] if args.skip_llm else [GGUF_MODELS[k] for k in args.llm]
    MODELS_DIR.mkdir(exist_ok=True)
    if args.source == "huggingface":
        from_huggingface(llms)
        return 0
    for name, (_, ref, pooling) in ST_MODELS.items():
        st_from_dockerhub(name, ref, pooling)
    for model in llms:
        gguf_from_dockerhub(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
