"""AI appraisal summary — condense an 80+ page appraisal to a few key pages.

Reads the uploaded appraisal PDF from valuation_documents, extracts its text
(PyMuPDF, pdfplumber fallback), and asks Claude for a structured summary:
how the appraiser approached the valuation, the key assumptions used, value
conclusions, market context, and risks. The result is stored per valuation
record (one summary per record, regenerating replaces it) and cross-checked
against the assumptions entered on the record.

Requires ANTHROPIC_API_KEY (same activation as the embedded AI assistant).
Scanned appraisals with no text layer are rejected with a clear message.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TEXT_CHARS = 350_000  # ~90k tokens of appraisal text, well inside context
MIN_TEXT_CHARS = 2_000    # below this the PDF is almost certainly scanned

_PROMPT = """You are a real estate valuation analyst. The text of a third-party
commercial real estate appraisal report follows. Condense it into a structured
summary a Valuation Committee can read in a few minutes instead of the full
report. Focus on HOW the appraiser approached the valuation and WHICH key
assumptions drive the value.

Return ONLY a JSON object (no markdown fences, no commentary) with exactly
these keys:

{
  "executive_summary": "3-6 sentence plain-English overview: what was appraised, the concluded value, how it moved vs the prior appraisal, and the one or two things that drive the number",
  "property": {"name": str, "location": str, "property_type": str, "gla_sf": number|null, "year_built": number|null, "occupancy_pct": number|null},
  "value_conclusion": {"as_is_value": number|null, "value_date": "YYYY-MM-DD"|null, "per_sf": number|null, "prior_value": number|null, "prior_value_date": str|null, "change_amount": number|null, "change_pct": number|null, "interest_appraised": str|null},
  "valuation_approach": "1-2 paragraphs: which approaches were developed (income/DCF, direct cap, sales comparison, cost), how they were weighted or reconciled, DCF hold period and reversion mechanics",
  "key_assumptions": {
    "overall_cap_rate": number|null, "terminal_cap_rate": number|null,
    "discount_rate": number|null, "market_rent_growth": number|null,
    "expense_growth": number|null, "real_estate_tax_growth": number|null,
    "general_inflation": number|null, "selling_costs_at_reversion": number|null,
    "vacancy_credit_loss": number|null, "dcf_hold_period_years": number|null,
    "notes": "anything unusual about how these were selected or supported (surveys, comps, prior appraisal)"
  },
  "in_place_income": {"noi_year_1": number|null, "effective_gross_revenue": number|null, "operating_expenses": number|null, "notes": str|null},
  "market_overview": ["3-6 bullets on the market/submarket conditions the appraiser relied on"],
  "rent_and_leasing": ["3-6 bullets: concluded market rents vs in-place, major tenants, rollover/renewal assumptions, downtime and TI/LC assumptions"],
  "positives": ["the appraiser's significant investment positives"],
  "risks": ["the appraiser's significant investment negatives/risks"],
  "extraordinary_assumptions": ["any extraordinary assumptions or hypothetical conditions, verbatim-ish; empty list if none"],
  "appraiser": {"firm": str|null, "appraisal_date": str|null, "report_type": str|null}
}

Rules:
- All rates as decimals (7.25 percent -> 0.0725). Dollar amounts as plain numbers.
- Use null when the report does not state a value. Never invent numbers.
- Quote the appraiser's reasoning where it explains an assumption, briefly.

APPRAISAL REPORT TEXT:
"""


def _extract_pdf_text(file_bytes: bytes) -> Tuple[str, int]:
    """Extract text from a PDF. Returns (text, page_count)."""
    try:
        import pymupdf
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages = len(doc)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts), pages
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed ({e}); falling back to pdfplumber")
        import io
        import pdfplumber
        parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = len(pdf.pages)
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts), pages


def generate_appraisal_summary(engine, record_id: int, username: str,
                               doc_id: Optional[int] = None) -> Dict[str, Any]:
    """Summarize the record's appraisal PDF via Claude and store the result."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("AI summary requires ANTHROPIC_API_KEY to be configured")

    with engine.connect() as conn:
        if doc_id:
            doc = conn.execute(text("""
                SELECT id, filename, file_data FROM valuation_documents
                WHERE id = :d AND record_id = :r
            """), {"d": doc_id, "r": record_id}).fetchone()
        else:
            doc = conn.execute(text("""
                SELECT id, filename, file_data FROM valuation_documents
                WHERE record_id = :r AND doc_type = 'appraisal' AND file_data IS NOT NULL
                ORDER BY created_at DESC LIMIT 1
            """), {"r": record_id}).fetchone()
    if not doc or not doc[2]:
        raise ValueError("No appraisal document uploaded for this valuation — upload the appraisal PDF first")

    doc_id, filename, file_bytes = int(doc[0]), doc[1], bytes(doc[2])

    pdf_text, page_count = _extract_pdf_text(file_bytes)
    if len(pdf_text.strip()) < MIN_TEXT_CHARS:
        raise ValueError(
            f"'{filename}' has little or no extractable text ({len(pdf_text.strip())} chars from "
            f"{page_count} pages) — it is likely a scanned image PDF. OCR it before summarizing."
        )
    truncated = len(pdf_text) > MAX_TEXT_CHARS
    if truncated:
        pdf_text = pdf_text[:MAX_TEXT_CHARS]

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": _PROMPT + pdf_text}],
    )
    response_text = message.content[0].text

    summary = _parse_json_object(response_text)
    summary["_meta"] = {
        "source_document": filename,
        "source_doc_id": doc_id,
        "page_count": page_count,
        "text_truncated": truncated,
        "model": MODEL,
    }

    payload = json.dumps(summary)
    from flask_app.services.valuation_service import _now
    with engine.begin() as conn:
        result = conn.execute(text("""
            UPDATE valuation_ai_summaries
            SET doc_id = :d, summary_json = :j, model = :m, created_by = :u, created_at = :now
            WHERE record_id = :r
        """), {"d": doc_id, "j": payload, "m": MODEL, "u": username, "now": _now(), "r": record_id})
        if result.rowcount == 0:
            conn.execute(text("""
                INSERT INTO valuation_ai_summaries
                    (record_id, doc_id, summary_json, model, created_by)
                VALUES (:r, :d, :j, :m, :u)
            """), {"r": record_id, "d": doc_id, "j": payload, "m": MODEL, "u": username})

    return get_ai_summary(engine, record_id)


def get_ai_summary(engine, record_id: int) -> Optional[Dict[str, Any]]:
    """Stored summary plus cross-checks against the record's entered assumptions."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT summary_json, model, created_by, created_at
            FROM valuation_ai_summaries WHERE record_id = :r
        """), {"r": record_id}).fetchone()
        rec = conn.execute(text("""
            SELECT concluded_value, cap_rate, term_cap_rate, discount_rate,
                   cost_of_sale_pct, direct_cap_noi
            FROM valuation_records WHERE id = :r
        """), {"r": record_id}).fetchone()
    if not row or not row[0]:
        return None

    summary = json.loads(row[0])
    checks: List[Dict[str, Any]] = []
    if rec is not None:
        ka = summary.get("key_assumptions") or {}
        vc = summary.get("value_conclusion") or {}
        ii = summary.get("in_place_income") or {}
        m = rec._mapping
        pairs = [
            ("Concluded Value", m["concluded_value"], vc.get("as_is_value"), 0.005),
            ("Going-in Cap Rate", m["cap_rate"], ka.get("overall_cap_rate"), 0.0001),
            ("Terminal Cap Rate", m["term_cap_rate"], ka.get("terminal_cap_rate"), 0.0001),
            ("Discount Rate", m["discount_rate"], ka.get("discount_rate"), 0.0001),
            ("Cost of Sale %", m["cost_of_sale_pct"], ka.get("selling_costs_at_reversion"), 0.0001),
            ("Direct Cap NOI", m["direct_cap_noi"], ii.get("noi_year_1"), 0.02),
        ]
        for label, entered, extracted, tol in pairs:
            if entered is None and extracted is None:
                continue
            match = None
            if entered is not None and extracted is not None:
                try:
                    e, x = float(entered), float(extracted)
                    match = abs(e - x) <= (tol * max(abs(e), abs(x)) if tol >= 0.005 else tol)
                except (TypeError, ValueError):
                    match = None
            checks.append({
                "field": label,
                "entered": entered,
                "extracted": extracted,
                "match": match,
            })

    return {
        "summary": summary,
        "checks": checks,
        "model": row[1],
        "created_by": row[2],
        "created_at": str(row[3]) if row[3] is not None else None,
    }


def _parse_json_object(response_text: str) -> Dict[str, Any]:
    """Parse the model's JSON object, tolerating markdown fences and any
    trailing commentary — decodes the first complete object it finds."""
    cleaned = response_text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence:
        cleaned = fence.group(1)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("AI summary returned no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned, start)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI summary returned invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise ValueError("AI summary returned JSON that is not an object")
    return obj
