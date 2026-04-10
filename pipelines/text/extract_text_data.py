#!/usr/bin/env python3
"""
extract_text_data.py
Handles structure: CompanyName / YearFolder / PDFs directly inside
Assigns recency weights to year folders automatically.
"""

import re
from pathlib import Path
from typing import Optional
import pdfplumber


def _parse_year_from_folder(folder_name: str) -> Optional[int]:
    """Extract ending year from folder names like FY 24-25, TATA_23-24, INFOYSIS_24."""
    m = re.search(r'(\d{2})-(\d{2})', folder_name)
    if m:
        return 2000 + int(m.group(2))
    m = re.search(r'[_\s](\d{2,4})$', folder_name)
    if m:
        y = int(m.group(1))
        return y if y > 2000 else 2000 + y
    m = re.search(r'(20\d{2})', folder_name)
    if m:
        return int(m.group(1))
    return None


def assign_year_weights(year_folders: list) -> dict:
    """Most recent year → 1.0, each older → -0.25, floor 0.1."""
    sorted_f = sorted(year_folders, key=lambda x: x[1], reverse=True)
    return {folder: round(max(1.0 - rank * 0.25, 0.1), 2)
            for rank, (folder, _) in enumerate(sorted_f)}


def _extract_quarter(filename: str, year: Optional[int]) -> str:
    m = re.search(r'[Qq]([1-4])', filename)
    if m and year:
        return f"{year}-Q{m.group(1)}"
    return f"{year}-Unknown" if year else "Unknown"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    lines = [l for l in text.splitlines()
             if not re.fullmatch(r'[\d\s\.\,\|\-\%\(\)]+', l.strip())
             and len(l.strip()) > 3]
    return "\n".join(lines)


def process_single_file(file_path: Path, company_name: str = "Unknown",
                         quarter: str = "Unknown", weight: float = 1.0) -> dict:
    suffix = file_path.suffix.lower()
    text = ""
    if suffix == ".pdf":
        try:
            with pdfplumber.open(file_path) as pdf:
                pages = [_clean_text(p.extract_text()) for p in pdf.pages]
            text = "\n".join(p for p in pages if p.strip())
        except Exception as e:
            return {"error": str(e)}
    elif suffix == ".txt":
        for enc in ("utf-8", "latin-1"):
            try:
                text = _clean_text(file_path.read_text(encoding=enc))
                break
            except Exception:
                continue
    if not text.strip():
        return {"error": "No text extracted"}
    return {"company": company_name, "quarter": quarter, "weight": weight,
            "source": "call_transcript", "raw_text": text, "filename": file_path.name}


def run_text_extraction_pipeline(companies_dir_path: str) -> list:
    """
    Scans: companies_dir/<Company>/<YearFolder>/*.pdf|txt
    Most recent year folder → weight 1.0, older → decreasing weights.
    """
    companies_dir = Path(companies_dir_path)
    if not companies_dir.exists():
        raise FileNotFoundError(f"Not found: {companies_dir}")

    records = []
    for company_dir in sorted(companies_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        company_name = company_dir.name

        # Collect year subfolders
        year_folders = []
        for sub in company_dir.iterdir():
            if not sub.is_dir():
                continue
            year = _parse_year_from_folder(sub.name)
            if year:
                year_folders.append((sub, year))

        # Fallback: PDFs directly in company folder
        if not year_folders:
            year_folders = [(company_dir, 2024)]

        weight_map = assign_year_weights(year_folders)

        for folder, year in year_folders:
            weight = weight_map.get(folder, 0.1)
            for file in sorted(folder.glob("*")):
                if file.suffix.lower() not in (".pdf", ".txt"):
                    continue
                result = process_single_file(
                    file, company_name, _extract_quarter(file.name, year), weight
                )
                if "error" not in result:
                    records.append(result)

    return records