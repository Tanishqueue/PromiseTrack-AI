#!/usr/bin/env python3
"""
extract_xbrl_data.py
Extracts XBRL/XML financial data from year-subfolder structure.
Handles: CompanyName / YearFolder / *.xml
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

METRIC_MAPPING = {
    "RevenueFromOperations": "revenue", "Revenue": "revenue",
    "Income": "revenue", "TotalIncome": "revenue",
    "ProfitLossForPeriod": "net_profit",
    "ProfitLossFromOrdinaryActivitiesAfterTax": "net_profit",
    "ProfitLossForPeriodFromContinuingOperations": "net_profit",
    "OperatingProfitBeforeProvisionAndContingencies": "operating_profit",
    "OperatingProfit": "operating_profit",
    "ProfitBeforeTax": "operating_profit",
    "ProfitBeforeExceptionalItemsAndTax": "operating_profit",
    "ProfitBeforeTaxAndExceptionalItems": "operating_profit",
    "Expenses": "total_expenses", "OperatingExpenses": "total_expenses",
    "ExpenditureExcludingProvisionsAndContingencies": "total_expenses",
    "TaxExpense": "tax_expense",
    "BasicEarningsPerShareAfterExtraordinaryItems": "eps",
}

# Tags that should NEVER be mapped to revenue even if they fuzzy-match "income".
# OtherIncome is a sub-component, not total revenue.
_REVENUE_BLOCKLIST = {
    "OtherIncome", "InterestEarned", "InterestOrDiscountOnAdvancesOrBills",
    "RevenueOnInvestments", "InterestOnBalancesWithReserveBankOfIndiaAndOtherInterBankFunds",
    "OtherInterest",
}

# Priority order for revenue: higher number = preferred when multiple tags map to "revenue".
# Only the highest-priority tag seen per (company, quarter, metric) is kept.
_METRIC_PRIORITY = {
    "revenue": {
        "Income": 100, "TotalIncome": 100,
        "RevenueFromOperations": 90, "Revenue": 90,
    },
    "net_profit": {
        "ProfitLossFromOrdinaryActivitiesAfterTax": 100,
        "ProfitLossForPeriod": 90,
        "ProfitLossForPeriodFromContinuingOperations": 80,
    },
    "operating_profit": {
        "OperatingProfitBeforeProvisionAndContingencies": 100,
        "OperatingProfit": 90,
        "ProfitBeforeExceptionalItemsAndTax": 80,
        "ProfitBeforeTaxAndExceptionalItems": 80,
        "ProfitBeforeTax": 70,
    },
}


def _parse_year_from_folder(folder_name: str) -> Optional[int]:
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


def _assign_weights(year_folders: list) -> dict:
    sorted_f = sorted(year_folders, key=lambda x: x[1], reverse=True)
    return {folder: round(max(1.0 - rank * 0.25, 0.1), 2)
            for rank, (folder, _) in enumerate(sorted_f)}


def _quarter_from_context(root, context_id: str) -> Optional[str]:
    try:
        context = root.find(f".//*[@id='{context_id}']")
        if context is None:
            return None
        period = context.find("{http://www.xbrl.org/2003/instance}period")
        if period is None:
            return None
        
        start_date = period.find("{http://www.xbrl.org/2003/instance}startDate")
        end_date = period.find("{http://www.xbrl.org/2003/instance}endDate")
        instant  = period.find("{http://www.xbrl.org/2003/instance}instant")
        
        # Ignore cumulative/YTD data (e.g., 9 months, 6 months). We only want ~90 day quarters.
        if start_date is not None and end_date is not None and start_date.text and end_date.text:
            from datetime import datetime
            try:
                sd = datetime.strptime(start_date.text, "%Y-%m-%d")
                ed = datetime.strptime(end_date.text, "%Y-%m-%d")
                days = (ed - sd).days
                if days > 105: # Skip periods longer than a quarter
                    return None
            except Exception:
                pass

        node = end_date if end_date is not None else instant
        if node is None or not node.text:
            return None
        m = re.match(r"(\d{4})-(\d{2})-\d{2}", node.text)
        if not m:
            return None
            
        year, month = int(m.group(1)), int(m.group(2))
        
        # ── FIX: Indian Financial Year Math ──
        # If the calendar month is Jan-Mar, the FY matches the calendar year.
        # If the calendar month is Apr-Dec, the FY is the NEXT calendar year.
        fy_year = year if month <= 3 else year + 1
        q = 4 if month <= 3 else (1 if month <= 6 else (2 if month <= 9 else 3))
        
        return f"{fy_year}-Q{q}"
        
    except Exception:
        return None


def _normalize_metric(tag: str) -> Optional[str]:
    clean = tag.split("}")[-1] if "}" in tag else tag.split(":")[-1]
    # Exact match first
    if clean in METRIC_MAPPING:
        # Block sub-components that fuzzy-match revenue but aren't total revenue
        if clean in _REVENUE_BLOCKLIST:
            return None
        return METRIC_MAPPING[clean]
    # Fuzzy match — but never let blocklisted tags through
    if clean in _REVENUE_BLOCKLIST:
        return None
    for key, val in METRIC_MAPPING.items():
        if key.lower() in clean.lower():
            return val
    return None


def _tag_priority(clean_tag: str, metric: str) -> int:
    """Return priority score for a tag within its metric group. Higher = preferred."""
    return _METRIC_PRIORITY.get(metric, {}).get(clean_tag, 50)


def _is_segment(context_id: str) -> bool:
    return any(x in context_id.lower()
               for x in ["segment", "reportablesegment", "geographicsegment"])


def extract_numeric_data(xml_path: Path, company_name: str,
                          weight: float = 1.0) -> list:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []

    # best[(quarter, metric)] = {"value": ..., "priority": ..., "weight": ...}
    best: dict = {}

    for elem in root.iter():
        if not (elem.get("unitRef") and elem.get("contextRef") and elem.text):
            continue
        try:
            value = float(elem.text)
        except (ValueError, TypeError):
            continue
        if abs(value) < 1:
            continue
        ctx = elem.get("contextRef")
        if _is_segment(ctx):
            continue
        quarter = _quarter_from_context(root, ctx)
        if not quarter:
            continue
        metric = _normalize_metric(elem.tag)
        if not metric:
            continue
        if abs(value) > 1_000_000:
            value = value / 10_000_000

        clean_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag.split(":")[-1]
        priority = _tag_priority(clean_tag, metric)
        key = (quarter, metric)

        if key not in best or priority > best[key]["priority"]:
            best[key] = {"value": round(value, 4), "priority": priority}

    return [
        {"company": company_name, "quarter": q, "metric": m,
         "value": v["value"], "weight": weight}
        for (q, m), v in best.items()
    ]


def run_xbrl_extraction_pipeline(companies_dir_path: str) -> list:
    """
    Scans: companies_dir/<Company>/<YearFolder>/*.xml
    Assigns recency weights same as text pipeline.
    """
    companies_dir = Path(companies_dir_path)
    if not companies_dir.exists():
        raise FileNotFoundError(f"Not found: {companies_dir}")

    all_records = []
    for company_dir in sorted(companies_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        company_name = company_dir.name

        year_folders = []
        for sub in company_dir.iterdir():
            if not sub.is_dir():
                continue
            year = _parse_year_from_folder(sub.name)
            if year:
                year_folders.append((sub, year))
        if not year_folders:
            year_folders = [(company_dir, 2024)]

        weight_map = _assign_weights(year_folders)

        for folder, year in year_folders:
            weight = weight_map.get(folder, 0.1)
            for xml_file in sorted(folder.glob("*.xml")):
                records = extract_numeric_data(xml_file, company_name, weight)
                all_records.extend(records)

    # Deduplicate
    seen, unique = set(), []
    for r in all_records:
        key = (r["company"], r["quarter"], r["metric"], r["value"])
        if key not in seen:
            unique.append(r)
            seen.add(key)
    unique.sort(key=lambda r: (r["company"], r["quarter"], r["metric"]))
    return unique