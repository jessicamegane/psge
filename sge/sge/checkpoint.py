"""Atomic rolling checkpoints for evolutionary experiments."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import platform
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


CHECKPOINT_VERSION = 1
CHECKPOINT_MAGIC = b"PSGE-CHECKPOINT\0"
CHECKSUM_SIZE = hashlib.sha256().digest_size
CHECKPOINT_PREFIX = "checkpoint_generation_"
CHECKPOINT_SUFFIX = ".pkl.gz"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely saved or restored."""


def _torch_module():
    """Return torch only when the experiment has already imported it."""
    return sys.modules.get("torch")


def seed_random_generators(seed: int) -> None:
    """Seed every random backend currently used by the process."""
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch_module()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch = _torch_module()
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError("Checkpoint contains invalid Python/NumPy RNG state") from exc

    if "torch_cpu" not in state and "torch_cuda" not in state:
        return

    torch = _torch_module()
    if torch is None:
        raise CheckpointError(
            "Checkpoint requires PyTorch RNG state, but PyTorch is not loaded"
        )
    try:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state:
            saved_cuda_states = state["torch_cuda"]
            if not torch.cuda.is_available():
                raise CheckpointError(
                    "Checkpoint requires CUDA RNG state, but CUDA is unavailable"
                )
            if len(saved_cuda_states) != torch.cuda.device_count():
                raise CheckpointError(
                    "Checkpoint CUDA device count does not match the current environment"
                )
            torch.cuda.set_rng_state_all(saved_cuda_states)
    except CheckpointError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError("Checkpoint contains invalid PyTorch RNG state") from exc


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameters_sha256(parameters: Dict[str, Any]) -> str:
    """Fingerprint resolved experiment settings, excluding recovery output paths."""
    ignored = {"RUN_FOLDER", "LOG_FOLDER", "RESUME_FROM"}

    def normalize(value):
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    normalized = {
        key: normalize(value)
        for key, value in parameters.items()
        if key not in ignored
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_metadata() -> Dict[str, Any]:
    torch = _torch_module()
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": getattr(torch, "__version__", None),
        "platform": platform.platform(),
    }


def _encode(payload: Dict[str, Any]) -> bytes:
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = gzip.compress(serialized)
    return CHECKPOINT_MAGIC + hashlib.sha256(compressed).digest() + compressed


def _decode(raw: bytes) -> Dict[str, Any]:
    header_size = len(CHECKPOINT_MAGIC) + CHECKSUM_SIZE
    if len(raw) <= header_size or not raw.startswith(CHECKPOINT_MAGIC):
        raise CheckpointError("Invalid checkpoint header")
    expected = raw[len(CHECKPOINT_MAGIC):header_size]
    compressed = raw[header_size:]
    if hashlib.sha256(compressed).digest() != expected:
        raise CheckpointError("Checkpoint checksum validation failed")
    try:
        payload = pickle.loads(gzip.decompress(compressed))
    except Exception as exc:
        raise CheckpointError("Checkpoint payload could not be decoded") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload is not a state dictionary")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            "Unsupported checkpoint version: %r" % payload.get("checkpoint_version")
        )
    required = {
        "completed_generation",
        "next_generation",
        "population",
        "previous_population",
        "best",
        "best_gen",
        "flag",
        "grammar_pcfg",
        "params",
        "rng_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointError("Checkpoint is missing fields: %s" % ", ".join(missing))
    return payload


def checkpoint_directory(run_folder: str) -> Path:
    return Path(run_folder) / "checkpoints"


def checkpoint_path(run_folder: str, completed_generation: int) -> Path:
    return checkpoint_directory(run_folder) / (
        f"{CHECKPOINT_PREFIX}{completed_generation:08d}{CHECKPOINT_SUFFIX}"
    )


def _checkpoint_candidates(directory: Path) -> Iterable[Path]:
    return sorted(
        directory.glob(f"{CHECKPOINT_PREFIX}*{CHECKPOINT_SUFFIX}"),
        reverse=True,
    )


def save_checkpoint(run_folder: str, state: Dict[str, Any], keep: int = 2) -> Path:
    """Atomically save state, then retain only the newest ``keep`` checkpoints."""
    if keep < 1:
        raise ValueError("At least one checkpoint must be retained")
    completed_generation = int(state["completed_generation"])
    payload = dict(state)
    payload["checkpoint_version"] = CHECKPOINT_VERSION
    payload.setdefault("runtime", runtime_metadata())
    encoded = _encode(payload)

    directory = checkpoint_directory(run_folder)
    directory.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_path(run_folder, completed_generation)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".checkpoint-", suffix=".tmp", dir=directory, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise CheckpointError(f"Could not save checkpoint {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    candidates = list(_checkpoint_candidates(directory))
    for obsolete in candidates[keep:]:
        try:
            obsolete.unlink()
        except OSError as exc:
            raise CheckpointError(
                f"Checkpoint saved, but obsolete checkpoint could not be removed: {obsolete}"
            ) from exc
    return destination


def load_checkpoint(path: str) -> Tuple[Dict[str, Any], Path]:
    """Load an exact file or the newest valid checkpoint in a run directory."""
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        candidates = [requested]
    else:
        directory = requested
        if directory.name != "checkpoints":
            directory = directory / "checkpoints"
        if not directory.is_dir():
            raise CheckpointError(f"Checkpoint directory does not exist: {directory}")
        candidates = list(_checkpoint_candidates(directory))

    if not candidates:
        raise CheckpointError(f"No checkpoints found for: {requested}")

    errors = []
    for candidate in candidates:
        try:
            payload = _decode(candidate.read_bytes())
            payload["checkpoint_file"] = str(candidate)
            return payload, candidate
        except (OSError, CheckpointError) as exc:
            errors.append(f"{candidate.name}: {exc}")
            if requested.is_file():
                break
    raise CheckpointError("No valid checkpoint found. " + "; ".join(errors))


def cleanup_checkpoints(run_folder: str) -> None:
    directory = checkpoint_directory(run_folder)
    if directory.exists():
        shutil.rmtree(directory)
