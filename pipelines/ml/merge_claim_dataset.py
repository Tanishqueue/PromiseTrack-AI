#!/usr/bin/env python3
"""
merge_claim_dataset.py
Merges claims + non-claims into a balanced, shuffled training dataset.
 
Flask entry point: run_merge_dataset_pipeline(claim_records, non_claim_records)
"""
 
import pandas as pd
 
 
def run_merge_dataset_pipeline(
    claim_records: list[dict],
    non_claim_records: list[dict],
) -> list[dict]:
    """
    Main entry point for the Flask app.
 
    Input : claim_records     — list of dicts with at least a 'sentence' key
                                (from run_claim_extraction_pipeline())
            non_claim_records — list of dicts with at least a 'sentence' key
                                (from run_non_claim_extraction_pipeline())
    Output: Deduplicated, shuffled list of dicts with keys: sentence, label.
    """
    claims             = pd.DataFrame(claim_records)
    non_claims         = pd.DataFrame(non_claim_records)
 
    claims["label"]     = "CLAIM"
    non_claims["label"] = "NON_CLAIM"
 
    df = pd.concat(
        [claims[["sentence", "label"]], non_claims[["sentence", "label"]]],
        ignore_index=True,
    )
 
    df = df.drop_duplicates(subset="sentence")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
 
    return df.to_dict(orient="records")
 