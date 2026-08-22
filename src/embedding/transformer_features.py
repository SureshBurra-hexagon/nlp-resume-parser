from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


@dataclass
class HuggingFaceTextEmbedder:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer dependencies are missing. Install `transformers` and `torch` to use phase 1 transformer models."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(  # type: ignore[operator]
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self._torch.no_grad():  # type: ignore[union-attr]
                outputs = self._model(**encoded)  # type: ignore[operator]
            token_embeddings = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            masked = token_embeddings * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            vectors.append(pooled.cpu().numpy())

        return np.vstack(vectors).astype(np.float32)
