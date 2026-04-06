from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoTokenizer

from nanoclustering.specter2.embedding_postprocess import EmbeddingPostprocessor, load_postprocessor

Pooling = Literal["cls", "mean"]

# SciRepEval/SPECTER2 control tokens (e.g., "[PRX]") are not part of the upstream
# `allenai/specter2_base` tokenizer vocab. We add them as special tokens and must
# resize token embeddings, which otherwise initializes new rows randomly.
# If we do not make that init deterministic, loading baseline/candidate in the
# same eval run yields different control-token embeddings and unstable metrics.
_CTRL_TOKEN_INIT_SEED = 42


@dataclass(frozen=True)
class LoadedAdapterModel:
    model: torch.nn.Module
    tokenizer: object
    postprocess: EmbeddingPostprocessor | None = None


def _require_adapters():
    try:
        from adapters import AutoAdapterModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: adapters.\n"
            "Recommended: use the Docker image in this repo:\n"
            "  docker build -t nc-scirepeval-adapters:latest docker/scirepeval_adapters\n"
            "Or install a pinned adapters stack (example):\n"
            "  pip install 'huggingface_hub==0.19.4' 'transformers==4.40.2' 'datasets==2.18.0' 'adapters==0.2.2'\n"
            "See `Materials/SPECTER2/scirepeval/` for the bundled training code."
        ) from exc
    return AutoAdapterModel


def _detect_adapter_source(adapter: str, adapter_source: str) -> str:
    if adapter_source in {"hf", "local"}:
        return adapter_source
    if adapter_source != "auto":
        raise ValueError("--adapter-source must be one of: auto|hf|local")
    if Path(adapter).exists():
        return "local"
    return "hf"


def load_adapter_model(
    base_model: str,
    *,
    adapter: str | None = None,
    adapter_name: str = "[PRX]",
    adapter_source: str = "auto",
    ctrl_tokens: list[str] | None = None,
    postprocess_artifact: str | None = None,
    device: str = "cpu",
) -> LoadedAdapterModel:
    AutoAdapterModel = _require_adapters()

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoAdapterModel.from_pretrained(base_model)

    if ctrl_tokens:
        # Deterministic token set/order (matches SciRepEval behavior).
        unique = sorted({str(t).strip() for t in ctrl_tokens if str(t).strip()})
        if unique:
            added = int(tokenizer.add_special_tokens({"additional_special_tokens": unique}))
            if added > 0:
                # Deterministic init for new token embedding rows.
                # Keep global RNG state unchanged for callers.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(_CTRL_TOKEN_INIT_SEED))
                    model.resize_token_embeddings(len(tokenizer))

    if adapter:
        src = _detect_adapter_source(adapter, adapter_source)
        if src == "hf":
            model.load_adapter(adapter, source="hf", load_as=adapter_name, set_active=True)
        else:
            model.load_adapter(adapter, load_as=adapter_name, set_active=True)
    else:
        # No adapter: use the base model as-is.
        pass

    model.to(torch.device(device))
    model.eval()
    postprocess = load_postprocessor(postprocess_artifact) if postprocess_artifact else None
    return LoadedAdapterModel(model=model, tokenizer=tokenizer, postprocess=postprocess)


def _pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling: Pooling,
    token_idx: int,
) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, token_idx]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
    raise ValueError(f"Unknown pooling: {pooling}")


def encode_texts(
    loaded: LoadedAdapterModel,
    texts: list[str],
    *,
    max_length: int = 512,
    batch_size: int = 32,
    pooling: Pooling = "cls",
    token_idx: int = 0,
    normalize: bool = True,
) -> torch.Tensor:
    device = next(loaded.model.parameters()).device
    out: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = loaded.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = loaded.model(**inputs, return_dict=True, output_hidden_states=True)
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                last_hidden_state = outputs.hidden_states[-1]
            pooled = _pool(
                last_hidden_state,
                inputs["attention_mask"],
                pooling=pooling,
                token_idx=token_idx,
            )
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            if loaded.postprocess is not None:
                pooled = loaded.postprocess.apply(pooled, normalize=normalize)
            out.append(pooled.detach().cpu())

    return torch.cat(out, dim=0) if out else torch.empty((0, 0))


def add_ctrl_token(texts: list[str], ctrl_token: str | None) -> list[str]:
    if not ctrl_token:
        return texts
    return [f"{ctrl_token} {t}" for t in texts]


def parse_pooling(value: str) -> Pooling:
    if value not in {"cls", "mean"}:
        raise argparse.ArgumentTypeError("--pooling must be 'cls' or 'mean'")
    return value  # type: ignore[return-value]
