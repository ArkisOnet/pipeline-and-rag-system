"""
Parser for services.xlsx — each row becomes one discrete Service Record.

No text chunking is applied: each row is treated as a single atomic unit
for the vector database.

Usage:
    from pipeline.parser.services_parser import load_service_records
    records = load_service_records("data/raw/services.xlsx")
    # → list of dicts with keys: text, source, service_code, icd_code, ...
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Expected column names in the xlsx
_COL_SERVICE_NAME = "Услуга"
_COL_SERVICE_CODE = "Код услуги"
_COL_ICD = "МКБ10"
_COL_DIAGNOSIS = "Диагноз"
_COL_PRICE = "Цена"
_COL_QUANTITY = "Количество"
_COL_CLAIMED = "Предъявленная сумма"
_COL_VISIT_REASON = "Повод обращения"
_COL_PLACE = "Место оказания услуги"
_COL_CDU = "Вид КДУ услуги"
_COL_DIAGNOSIS_TYPE = "Типа диагноза"
_COL_PERIOD = "Период услуги"


def _safe_str(val) -> str:
    """Convert a cell value to string, returning '' for NaN/None."""
    import math
    if val is None:
        return ""
    try:
        if math.isnan(float(val)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def load_service_records(path: str) -> list[dict]:
    """
    Read services.xlsx and return one dict per row.

    Each dict has:
        text      — combined string used for embedding (Услуга + МКБ10 + Диагноз)
        source    — always "service"
        + all metadata fields as flat keys
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas not installed. Run: pip install pandas openpyxl") from exc

    xlsx_path = Path(path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Services file not found: {xlsx_path}")

    logger.info("Reading services file: %s", xlsx_path)
    df = pd.read_excel(xlsx_path)
    logger.info("Loaded %d service rows", len(df))

    records: list[dict] = []
    for idx, row in df.iterrows():
        service_name = _safe_str(row.get(_COL_SERVICE_NAME))
        icd_code = _safe_str(row.get(_COL_ICD))
        diagnosis = _safe_str(row.get(_COL_DIAGNOSIS))

        # Primary text for embedding: combine the three key fields
        parts = [p for p in [service_name, icd_code, diagnosis] if p]
        text = " | ".join(parts)

        if not text:
            continue  # skip fully empty rows

        records.append({
            "text": text,
            "source": "service",
            "service_name": service_name,
            "service_code": _safe_str(row.get(_COL_SERVICE_CODE)),
            "icd_code": icd_code,
            "diagnosis": diagnosis,
            "price": _safe_float(row.get(_COL_PRICE)),
            "quantity": _safe_int(row.get(_COL_QUANTITY)),
            "claimed_amount": _safe_float(row.get(_COL_CLAIMED)),
            "visit_reason": _safe_str(row.get(_COL_VISIT_REASON)),
            "place_of_service": _safe_str(row.get(_COL_PLACE)),
            "cdu_type": _safe_str(row.get(_COL_CDU)),
            "diagnosis_type": _safe_str(row.get(_COL_DIAGNOSIS_TYPE)),
            "period": _safe_str(row.get(_COL_PERIOD)),
        })

    logger.info("Parsed %d valid service records", len(records))
    return records
