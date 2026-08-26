"""Local sentence embedding adapter for communication entropy."""

from __future__ import annotations

import numpy as np


class LocalTransformerEmbedder:
    def __init__(self, model_path: str, max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    def __call__(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().numpy()
