#!/usr/bin/env python3
"""
run_claim_model.py
Inference pipeline: loads trained DistilBERT claim classifier and
runs it on sentence records passed in from the Flask service layer.

Flask entry point : run_claim_model_pipeline(sentence_records)
Model loading     : call load_claim_model(model_path) once at app startup
                    (stored as a module-level singleton — never reloaded)
"""

import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Inference config ──────────────────────────────────────────────────────────

BATCH_SIZE = 64
THRESHOLD  = 0.6
MAX_LENGTH = 128

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Module-level singletons ───────────────────────────────────────────────────

_tokenizer: Optional[AutoTokenizer] = None
_model:     Optional[AutoModelForSequenceClassification] = None


# ── Model loader (call once at app startup) ───────────────────────────────────

def load_claim_model(model_path: str) -> None:
    """
    Loads the DistilBERT tokenizer and model into module-level singletons.
    Call this once inside create_app() — NOT inside a request handler.

    Args:
        model_path: Absolute path to the model checkpoint directory.
                    Use config.MODEL_PATH to supply this.
    """
    global _tokenizer, _model

    if _model is not None:
        return  # Already loaded — skip silently

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    _model.to(DEVICE)
    _model.eval()


def is_model_loaded() -> bool:
    return _model is not None


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(sentences: list[str]) -> list[dict]:
    """
    Runs batched inference on a list of sentence strings.
    Returns list of {label, confidence} dicts in the same order.
    """
    id2label  = _model.config.id2label
    claim_idx = next((k for k, v in id2label.items() if v.upper() == "CLAIM"), 1)
    softmax   = torch.nn.Softmax(dim=-1)
    results   = []

    for start in range(0, len(sentences), BATCH_SIZE):
        batch   = sentences[start : start + BATCH_SIZE]
        encoded = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            logits = _model(**encoded).logits

        probs = softmax(logits).cpu()

        for row in probs:
            claim_prob = row[claim_idx].item()
            results.append({
                "label":      "CLAIM" if claim_prob >= THRESHOLD else "NON_CLAIM",
                "confidence": round(claim_prob, 4),
            })

    return results


# ── Flask entry point ─────────────────────────────────────────────────────────

def run_claim_model_pipeline(sentence_records: list[dict]) -> list[dict]:
    """
    Main entry point for the Flask app.

    Input : List of dicts from run_sentence_splitting_pipeline(),
            each must have: sentence, company, quarter
    Output: Filtered list of dicts classified as CLAIM by the model,
            deduplicated, with confidence score attached.

    Raises RuntimeError if load_claim_model() has not been called yet.
    """
    if not is_model_loaded():
        raise RuntimeError(
            "Model is not loaded. Call load_claim_model(config.MODEL_PATH) "
            "inside create_app() before handling requests."
        )

    # Clean and filter empty sentences
    cleaned = [
        {**r, "sentence": _clean(str(r.get("sentence", "")))}
        for r in sentence_records
    ]
    cleaned = [r for r in cleaned if len(r["sentence"]) > 0]

    if not cleaned:
        return []

    preds = _run_inference([r["sentence"] for r in cleaned])

    # Attach predictions and filter to CLAIM only
    results = []
    seen    = set()

    for record, pred in zip(cleaned, preds):
        if pred["label"] != "CLAIM":
            continue
        key = record["sentence"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "company":    record.get("company", ""),
            "quarter":    record.get("quarter", ""),
            "sentence":   record["sentence"],
            "label":      pred["label"],
            "confidence": pred["confidence"],
        })

    return results