#!/usr/bin/env python3
"""
split_sentences.py
Splits raw transcript text into clean sentences using nltk.sent_tokenize.
Output: sentence_data.csv  (company, quarter, source, sentence)
"""

import re
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

# --- Resource Initialization ---
def _ensure_nltk_resources():
    """Quietly ensures necessary NLTK models are downloaded."""
    for resource in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1], quiet=True)

_ensure_nltk_resources()

# --- Garbage Patterns (Constants) ---
_GARBAGE_RE = re.compile(
    r'^[\d\s\.\,\|\-\%\(\)\$\₹\/\:]+$'   # purely numeric / symbolic
    r'|^\s*[A-Z\s]{1,6}\s*$'              # ALL-CAPS short header (≤6 words)
    r'|www\.|http|©|™|®',                 # URLs / legal symbols
    re.IGNORECASE,
)

# --- Internal Helpers ---

def _clean_sentence(s: str) -> str:
    """Normalize whitespace and strip newlines."""
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def _is_valid_sentence(s: str) -> bool:
    """Return True if the sentence is worth keeping (length/content check)."""
    if len(s) < 20:
        return False
    if _GARBAGE_RE.search(s):
        return False
    # Must contain at least three alphabetic words
    words = re.findall(r'[a-zA-Z]{2,}', s)
    return len(words) >= 3

# --- Core Callable Functions ---

def process_text_into_sentences(text: str) -> list[str]:
    """
    Takes a raw string and returns a list of cleaned, validated sentences.
    Useful for processing single inputs from a web form.
    """
    if not text or pd.isna(text):
        return []
    
    raw_sentences = sent_tokenize(str(text))
    valid_sentences = []
    
    for s in raw_sentences:
        cleaned = _clean_sentence(s)
        if _is_valid_sentence(cleaned):
            valid_sentences.append(cleaned)
            
    return valid_sentences

def run_sentence_splitting_pipeline(extracted_records: list[dict]) -> list[dict]:
    """
    The main entry point for the batch pipeline.
    Input: List of dicts from 'run_text_extraction_pipeline'.
    Output: List of dicts where each record is a single sentence.
    """
    sentence_records = []
    
    for record in extracted_records:
        raw_text = record.get("raw_text", "")
        sentences = process_text_into_sentences(raw_text)
        
        for s in sentences:
            sentence_records.append({
                "company":  record.get("company"),
                "quarter":  record.get("quarter"),
                "source":   record.get("source"),
                "sentence": s,
            })

    # Drop duplicates naturally in the list before returning
    seen = set()
    unique_records = []
    for rec in sentence_records:
        if rec["sentence"] not in seen:
            unique_records.append(rec)
            seen.add(rec["sentence"])
            
    return unique_records