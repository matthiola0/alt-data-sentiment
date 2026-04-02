"""FinBERT (`ProsusAI/finbert`) CPU batch wrapper.

One FinBERT forward pass per document; the caller is responsible for
fanning the resulting (pos, neg, neu) triple out to every ticker linked
to that document. Texts longer than 512 tokens are truncated.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "ProsusAI/finbert"
# FinBERT label order in the released model
LABEL_ORDER = ["positive", "negative", "neutral"]


def load_finbert(
    cache_dir: str | Path | None = None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load tokenizer + model on CPU, eval mode."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, cache_dir=cache_dir
    )
    model.eval()
    model.to("cpu")
    # Align our LABEL_ORDER to the model's id2label in case HF reorders later
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    # Build a permutation so columns come out [pos, neg, neu] regardless of
    # the model's internal ordering
    perm = [next(i for i, lbl in id2label.items() if lbl.lower().startswith(want[:3]))
            for want in LABEL_ORDER]
    model._finbert_perm = perm  # attach for score_batch
    return tok, model


def score_batch(
    texts: Iterable[str],
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """Score an iterable of texts. Returns array shape (n, 3) in the order
    [positive, negative, neutral]. Empty / non-string inputs become NaN rows.
    """
    texts = list(texts)
    n = len(texts)
    out = np.full((n, 3), np.nan, dtype=np.float32)

    perm = getattr(model, "_finbert_perm", [0, 1, 2])

    # Identify indices with usable text
    usable_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not usable_idx:
        return out

    with torch.inference_mode():
        for bstart in range(0, len(usable_idx), batch_size):
            batch_ix = usable_idx[bstart : bstart + batch_size]
            batch_texts = [texts[i] for i in batch_ix]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Reorder columns to [pos, neg, neu]
            probs = probs[:, perm]
            for row_i, dst in enumerate(batch_ix):
                out[dst] = probs[row_i]

    return out
