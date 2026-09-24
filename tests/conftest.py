import os

import numpy as np
import pytest
import soundfile as sf

from cohere_transcribe_ara_cli.testing import build_synthetic_dataset, build_tiny_model


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """Tiny random model + synthetic dataset (built once per session)."""
    root = tmp_path_factory.mktemp("tiny")
    model_dir = build_tiny_model(str(root))
    train, valid = build_synthetic_dataset(str(root), n_train=24, n_valid=6)
    return {"root": str(root), "model": model_dir, "train": train, "valid": valid}


@pytest.fixture
def wav_factory(tmp_path):
    def make(name: str, seconds: float, sr: int = 16000, channels: int = 1) -> str:
        path = os.path.join(tmp_path, name)
        y = 0.1 * np.random.default_rng(0).standard_normal((int(seconds * sr), channels)).astype(np.float32)
        sf.write(path, y if channels > 1 else y[:, 0], sr)
        return path

    return make
