"""
Pytest configuration — stubs heavyweight ML dependencies so the full test
suite runs without downloading models (~500 MB).

In CI and production the real libraries are installed; the stubs are only
used when a library is absent, so they never mask real bugs there.
"""
import sys
import types

import numpy as np


def _stub_sentence_transformers() -> None:
    """Register a minimal sentence-transformers stub if the real one is absent."""
    if "sentence_transformers" in sys.modules:
        return  # real library already imported — don't override

    try:
        import sentence_transformers  # noqa: F401
        return  # real library available
    except ModuleNotFoundError:
        pass

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self._dim = 384

        def encode(
            self,
            texts,
            normalize_embeddings: bool = True,
            convert_to_numpy: bool = True,
            show_progress_bar: bool = False,
        ) -> np.ndarray:
            n = len(texts) if isinstance(texts, list) else 1
            vecs = np.random.rand(n, self._dim).astype("float32")
            if normalize_embeddings:
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = mod


# Run once per session before any test is collected.
_stub_sentence_transformers()
