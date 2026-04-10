#!/usr/bin/env python3
"""
non_claim_extractor.py
 
1. Removes sentences already identified as claims from the pool.
2. Runs a lightweight second-pass keyword score on the remainder.
3. Sentences scoring above threshold are rescued into claims
   (catches what the spaCy pass missed).
4. Samples the clean remainder to balance the dataset.
 
Flask entry point: run_non_claim_extraction_pipeline(sentence_records, claim_records)
"""
"""
import re
import pandas as pd
 
 
# ── Keyword sets for second-pass scoring ──────────────────────────────────────
 
_PERF = {
    "revenue", "revenues", "sales", "profit", "profits", "ebitda", "margin",
    "margins", "income", "earnings", "growth", "decline", "volume", "volumes",
    "cost", "costs", "loan", "loans", "deposit", "deposits", "credit",
    "arpu", "aum", "subscriber", "subscribers", "customer", "customers",
    "order", "orders", "capacity", "utilisation", "utilization", "share",
    "contribution", "mix", "demand", "supply", "price", "pricing", "return",
    "cash", "debt", "capex", "provision", "coverage", "ratio", "fee", "spread",
    "collection", "disbursement", "addition", "base", "book", "output",
}
 
_DIR = {
    "grew", "grow", "growth", "increase", "increased", "rise", "rose",
    "improve", "improved", "improvement", "expand", "expanded", "expansion",
    "decline", "declined", "decrease", "decreased", "fall", "fell", "drop",
    "strong", "robust", "healthy", "solid", "stable", "steady", "record",
    "higher", "lower", "better", "significant", "substantial", "momentum",
    "recover", "recovered", "deliver", "delivered", "achieve", "achieved",
    "surge", "surged", "jump", "jumped", "scale", "scaled", "accelerat",
    "well", "good", "broad", "double", "triple", "outperform",
}
 
_NEGATION_RE   = re.compile(r"\b(not|no|never|don't|do not|didn't|did not|won't|will not)\b", re.IGNORECASE)
_QUESTION_RE   = re.compile(r"\?$")
_GARBAGE_RE    = re.compile(r"^[\d\s\.\,\%\|\-\(\)\$\₹\/\:]+$")
_TRANSITION_RE = re.compile(
    r"^(thank|thanks|good (morning|evening|afternoon)|hi |hello |"
    r"let me (first|now|just)|moving on|turning to|coming to|over to|"
    r"operator|moderator|so (let|shall) (me|us))",
    re.IGNORECASE,
)
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _dedup_key(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\s+", " ", str(s).lower())).strip()
 
def keyword_score(sentence: str) -> int:
    
    s      = sentence.lower()
    tokens = set(re.findall(r"[a-z]+", s))
    perf_hits = sum(1 for kw in _PERF if kw in s)
    dir_hits  = sum(1 for kw in _DIR  if any(t.startswith(kw) for t in tokens))
    return perf_hits + dir_hits
 
def is_likely_claim(sentence: str) -> bool:
   
    s = sentence.strip()
    if len(s) < 25:
        return False
    if _QUESTION_RE.search(s) or _GARBAGE_RE.match(s) or _TRANSITION_RE.match(s):
        return False
    if _NEGATION_RE.search(s):
        return False
    return keyword_score(s) >= 3
 
 
# ── Flask entry point ─────────────────────────────────────────────────────────
 
def run_non_claim_extraction_pipeline(
    sentence_records: list[dict],
    claim_records: list[dict],
) -> dict:
    
        
    sentences_df = pd.DataFrame(sentence_records)
    claims_df    = pd.DataFrame(claim_records)
 
    # Build exclusion set from existing claims
    claim_keys             = set(claims_df["sentence"].apply(_dedup_key))
    sentences_df["_key"]   = sentences_df["sentence"].apply(_dedup_key)
 
    remainder = (
        sentences_df[~sentences_df["_key"].isin(claim_keys)]
        .drop_duplicates("_key")
        .copy()
    )
 
    # Second-pass: rescue missed claims from the remainder
    rescued_mask = remainder["sentence"].apply(is_likely_claim)
    rescued      = remainder[rescued_mask].drop(columns="_key")
    true_non     = remainder[~rescued_mask].drop(columns="_key")
 
    # Merge rescued into claims, deduplicate
    all_claims_df = pd.concat([claims_df, rescued], ignore_index=True)
    all_claims_df["_key"] = all_claims_df["sentence"].apply(_dedup_key)
    all_claims_df = (
        all_claims_df.drop_duplicates("_key")
                     .drop(columns="_key")
                     .reset_index(drop=True)
    )
 
    # Sample non-claims to match claim count (balanced dataset)
    true_non  = true_non.drop_duplicates(subset="sentence")
    n_sample  = min(len(all_claims_df), len(true_non))
    non_claims_df = true_non.sample(n=n_sample, random_state=42).copy()
    non_claims_df["label"] = "NON_CLAIM"
 
    return {
        "claims":     all_claims_df.to_dict(orient="records"),
        "non_claims": non_claims_df.to_dict(orient="records"),
    }
"""
    
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT - specifically tuned for sentiment & claim-like financial tones
# For a production agent, you'd use a local model to save on API costs
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def is_numeric_claim(sentence: str) -> bool:
    """
    Core logic: A financial claim must contain numeric proof 
    (dates, percentages, currency, or counts).
    """
    # Regex for: $10M, 5%, 2026, 1.5bn, ₹500cr
    numeric_pattern = r"(\d+\.?\d*)\s?([%m|bn|cr|k|%|₹|\$])"
    return bool(re.search(numeric_pattern, sentence, re.IGNORECASE))

def get_finbert_score(sentence: str):
    """Uses Transformer to check if the sentence has a 'factual' or 'positive/negative' tone."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 0: Positive, 1: Negative, 2: Neutral
    # Claims are rarely 'Neutral'—they are usually driving a narrative (Pos/Neg)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    conf, label = torch.max(probs, dim=-1)
    return label.item(), conf.item()

def run_non_claim_extraction_pipeline(sentence_records, claim_records):
    """
    Main entry point - same I/O as your old code, but with deep learning logic.
    """
    sentences_df = pd.DataFrame(sentence_records)
    claims_df    = pd.DataFrame(claim_records)
    
    # 1. First Pass: Numerical Density Filter
    # Most management claims contain a hard number (Numeric Drift Detection)
    sentences_df["is_claim_candidate"] = sentences_df["sentence"].apply(is_numeric_claim)
    
    # 2. Second Pass: FinBERT Validation
    # We only run the heavy transformer on candidates to save compute
    def validate_claim(row):
        if not row["is_claim_candidate"]:
            return False
        label, conf = get_finbert_score(row["sentence"])
        # We accept sentences where FinBERT is confident it's NOT just neutral chatter
        return label != 2 and conf > 0.85

    sentences_df["is_verified_claim"] = sentences_df.apply(validate_claim, axis=1)
    
    claims_df = sentences_df[sentences_df["is_verified_claim"]].copy()
    non_claims_df = sentences_df[~sentences_df["is_verified_claim"]].copy()
    
    # Labeling for your balanced dataset
    claims_df["label"] = "CLAIM"
    non_claims_df["label"] = "NON_CLAIM"
    
    # Balancing the dataset (Sample non-claims to match claims count)
    n_sample = min(len(claims_df), len(non_claims_df))
    balanced_non_claims = non_claims_df.sample(n=n_sample, random_state=42)
    
    return {
        "claims": claims_df.to_dict(orient="records"),
        "non_claims": balanced_non_claims.to_dict(orient="records")
    }