import logging
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class DescriptionEmbedder:
    """Encode account descriptions into embeddings using a transformer model."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dim = self.model.config.hidden_size
        logger.info("Loaded transformer model %s", model_name)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Return embeddings for a batch of texts."""
        if not texts:
            return np.zeros((0, self.dim))

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings
