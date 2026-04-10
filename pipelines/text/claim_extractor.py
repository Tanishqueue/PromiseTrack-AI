#!/usr/bin/env python3
"""
claim_extractor.py

Generalised CLAIM extractor using spaCy linguistic structure.

Strategy (works for ANY earnings call transcript):
  A sentence is a CLAIM if it describes business/financial performance.
  Detection is based on:
    1. Dependency parse: financial noun as subject of a directional verb
    2. Named entity signals: MONEY, PERCENT, CARDINAL near financial terms
    3. Keyword density scoring as a fallback (broad, not company-specific)
    4. Structural exclusions: questions, greetings, transitions, pure Q&A lines

  We deliberately avoid hard-coding company names, product names, or
  transcript-specific patterns so the extractor generalises to any company.
"""
import re
import spacy
import pandas as pd
from pathlib import Path

# Load spaCy model once at the module level
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ── Financial domain vocabulary ───────────────────────────────────────────────

FINANCIAL_NOUNS = {
    "revenue", "revenues", "sales", "turnover", "income", "earnings",
    "profit", "profits", "loss", "losses", "ebitda", "ebit", "pat", "pbt",
    "margin", "margins", "cost", "costs", "expense", "expenses", "expenditure",
    "capex", "opex", "cashflow", "cash", "debt", "leverage",
    "return", "yield", "dividend", "buyback",
    "volume", "volumes", "growth", "decline", "performance",
    "loan", "loans", "deposit", "deposits", "credit", "npa", "gnpa", "nnpa",
    "arpu", "aum", "subscriber", "subscribers", "customer", "customers",
    "order", "orders", "backlog", "capacity", "utilisation", "utilization",
    "share", "market share", "segment", "portfolio", "balance sheet",
    "provision", "coverage", "ratio", "rate", "basis point", "bps",
    "contribution", "mix", "composition", "proportion", "penetration",
    "addition", "accretion", "disbursement", "disbursements",
    "collection", "collections", "recovery", "recoveries",
    "number", "count", "base", "book", "index", "indices",
    "demand", "supply", "output", "throughput",
    "price", "pricing", "realization", "realisation",
    "fee", "fees", "commission", "spread", "net interest",
}

DIRECTIONAL_VERBS = {
    "grow", "grew", "increase", "increased", "rise", "rose", "improve",
    "improved", "expand", "expanded", "decline", "declined", "decrease",
    "decreased", "fall", "fell", "drop", "dropped", "compress", "compressed",
    "recover", "recovered", "remain", "remained", "sustain", "sustained",
    "deliver", "delivered", "achieve", "achieved", "outperform", "outperformed",
    "surge", "surged", "jump", "jumped", "moderate", "moderated",
    "stabilise", "stabilize", "stabilised", "stabilized",
    "accelerate", "accelerated", "decelerate", "decelerated",
    "widen", "widened", "narrow", "narrowed", "strengthen", "strengthened",
    "weaken", "weakened", "scale", "scaled",
}

FINANCIAL_ADJECTIVES = {
    "strong", "robust", "healthy", "solid", "stable", "steady",
    "weak", "lower", "higher", "better", "worse", "significant",
    "record", "resilient", "positive", "negative", "flat",
    "elevated", "compressed", "improved", "declining", "growing",
}

NON_CLAIM_ROOT_VERBS = {
    "say", "think", "believe", "want", "plan", "hope",
    "ask", "tell", "mention", "note", "highlight", "explain",
    "discuss", "refer", "suggest", "wish", "congratulate", "welcome",
    "introduce", "hand", "open", "proceed", "begin", "start", "end", "close",
    "request", "invite", "take", "go", "come", "move", "turn",
}

BE_VERBS = {"be", "is", "are", "was", "were", "am"}


# ── Regex patterns ────────────────────────────────────────────────────────────

_TRANSITION_RE = re.compile(
    r'^(thank|thanks|good (morning|evening|afternoon)|congratulation|'
    r'hi |hello |let me (first|now|just)|i will now|moving on|'
    r'turning to|coming to|next (up|is|we)|over to|back to you|'
    r'operator|moderator|so (let|shall) (me|us)|'
    r'before (i|we) (go|move|turn)|just to (clarify|confirm|add))',
    re.IGNORECASE,
)

_GARBAGE_RE = re.compile(r'^[\d\s\.\,\%\|\-\(\)\$\₹\/\:]+$')

_ARTEFACT_RE = re.compile(
    r'\s*(Classification\s*[-–]\s*\w+|Page\s+\d+\s+of\s+\d+|'
    r'Confidential|Internal Use Only|Public)\s*.*$',
    re.IGNORECASE,
)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _has_financial_entity(doc) -> bool:
    return any(ent.label_ in {"MONEY", "PERCENT", "CARDINAL"} for ent in doc.ents)

def _financial_noun_score(doc) -> int:
    return sum(1 for t in doc if t.lemma_.lower() in FINANCIAL_NOUNS)

def _directional_verb_score(doc) -> int:
    return sum(1 for t in doc if t.lemma_.lower() in DIRECTIONAL_VERBS)

def _financial_adj_score(doc) -> int:
    return sum(1 for t in doc if t.pos_ == "ADJ" and t.lemma_.lower() in FINANCIAL_ADJECTIVES)

def _subject_is_financial(doc) -> bool:
    return any(
        t.dep_ in {"nsubj", "nsubjpass"} and t.lemma_.lower() in FINANCIAL_NOUNS
        for t in doc
    )

def _root_lemma(doc) -> str:
    return next((t.lemma_.lower() for t in doc if t.dep_ == "ROOT"), "")

def _has_negation(doc) -> bool:
    return any(t.dep_ == "neg" for t in doc)


# ── Core classifier ───────────────────────────────────────────────────────────

def is_claim(sentence: str, doc=None) -> bool:
    """
    Returns True if the sentence is a financial performance claim.
    If doc is provided, skips NLP processing (efficient for batch use via nlp.pipe).
    """
    s = sentence.strip()

    # Hard exclusions
    if s.endswith("?") or _TRANSITION_RE.match(s) or _GARBAGE_RE.match(s) or len(s) < 25:
        return False

    if doc is None:
        doc = nlp(s)

    root = _root_lemma(doc)

    # Negated sentences are not claims
    if _has_negation(doc):
        return False

    # Speech/intent root with no financial entity = not a claim
    if root in NON_CLAIM_ROOT_VERBS and not _has_financial_entity(doc):
        return False

    # Compute signals
    fn_score   = _financial_noun_score(doc)
    dv_score   = _directional_verb_score(doc)
    adj_score  = _financial_adj_score(doc)
    has_entity = _has_financial_entity(doc)
    subj_fin   = _subject_is_financial(doc)
    root_dir   = root in DIRECTIONAL_VERBS
    root_be    = root in BE_VERBS

    # `be`-root sentences need stronger evidence
    if root_be:
        if fn_score >= 1 and has_entity:   # e.g. "EBITDA is ₹18,000 crores"
            return True
        if subj_fin and adj_score >= 1:    # e.g. "margins are stable"
            return True
        return False

    # Strong structural signals
    if subj_fin and root_dir:              # financial subject + directional verb
        return True
    if fn_score >= 1 and dv_score >= 1:   # financial noun + directional verb anywhere
        return True
    if fn_score >= 1 and has_entity:       # financial noun + numeric/money entity
        return True
    if (fn_score + dv_score + adj_score) >= 4:  # high keyword density fallback
        return True
    if fn_score >= 1 and adj_score >= 1:   # financial noun + quality adjective
        return True

    return False


# ── Text cleaning ─────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return _ARTEFACT_RE.sub('', s).strip()

def _dedup_key(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]', ' ', re.sub(r'\s+', ' ', s.lower())).strip()


# ── Flask app entry point ─────────────────────────────────────────────────────

def run_claim_extraction_pipeline(sentence_records: list[dict]) -> list[dict]:
    """
    Main entry point for the Flask app.

    Input : List of sentence dicts (from split_sentences.py), each must have
            at least a "sentence" key. Any extra keys (company, quarter, etc.)
            are preserved in the output.
    Output: Deduplicated list of dicts classified as financial claims.
    """
    # Normalise text in place on a working copy
    sentences = [normalize_text(str(r["sentence"])) for r in sentence_records]

    results = []
    for i, doc in enumerate(nlp.pipe(sentences, batch_size=256)):
        if is_claim(sentences[i], doc):
            record = sentence_records[i].copy()
            record["sentence"] = sentences[i]
            results.append(record)

    # Deduplicate on normalised sentence content
    seen = set()
    unique_claims = []
    for r in results:
        key = _dedup_key(r["sentence"])
        if key not in seen:
            unique_claims.append(r)
            seen.add(key)

    return unique_claims