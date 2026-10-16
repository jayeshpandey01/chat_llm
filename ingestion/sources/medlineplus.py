"""NIH MedlinePlus authoritative medical knowledge and web scraping ingestion engine.

Fetches clinical summaries, disease overviews, pathognomonic symptoms, and when to seek
emergency care from NIH / National Library of Medicine (NLM) MedlinePlus.
Features seamless local FTS5 SQLite fallback for offline and sandboxed environments.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx
from loguru import logger

from core.schemas import ScrapedDocumentSchema
from ingestion.sources.base import BaseIngestionSource, canonicalize_url

MEDLINEPLUS_SEARCH_API = "https://wsearch.nlm.nih.gov/ws/query"
MEDLINEPLUS_WEB_BASE = "https://medlineplus.gov"
USER_AGENT = "HealthTrackAI-ClinicalIngestion/1.0 (NIH-MedlinePlus-Compliant; contact: safety@healthtrack.ai)"


class MedlinePlusSource(BaseIngestionSource):
    """Authoritative medical source scraping and querying NIH / NLM MedlinePlus."""

    def __init__(self, db_path: Optional[Path] = None, timeout: float = 8.0):
        super().__init__(name="medlineplus")
        self.timeout = timeout
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.db_path = db_path or (base_dir / "data" / "processed" / "clinical_evidence.db")

    async def fetch(self, query: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Fetch MedlinePlus articles matching clinical query.

        Attempts live HTTP API query first; falls back to local SQLite FTS5 evidence store.
        """
        results: list[ScrapedDocumentSchema] = []
        clean_query = query.strip()
        if not clean_query:
            return results

        # 1. Attempt live HTTP web search
        live_docs = await self._fetch_live(clean_query, max_results)
        if live_docs:
            return live_docs

        # 2. Resilient local NIH / MedlinePlus SQLite FTS5 fallback
        return self._fetch_local_fallback(clean_query, max_results)

    async def _fetch_live(self, query: str, max_results: int) -> list[ScrapedDocumentSchema]:
        """Query NIH MedlinePlus web service API or direct HTML topic pages."""
        headers = {"User-Agent": USER_AGENT}
        params = {
            "db": "healthTopics",
            "term": query,
            "retmode": "json",
            "rettype": "brief",
            "max": str(max_results),
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                resp = await client.get(MEDLINEPLUS_SEARCH_API, params=params)
                if resp.status_code == 200:
                    docs = []
                    # 1. Parse standard NLM XML search return
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(resp.text)
                        for doc_elem in root.findall(".//document")[:max_results]:
                            url = doc_elem.attrib.get("url", f"{MEDLINEPLUS_WEB_BASE}/healthtopics.html")
                            canon_url = canonicalize_url(url) or url
                            title = ""
                            snippet = ""
                            for c in doc_elem.findall("content"):
                                name = c.attrib.get("name", "")
                                text = re.sub(r"<[^>]+>", "", c.text or "").strip()
                                if name == "title":
                                    title = text
                                elif name in ("FullSummary", "snippet"):
                                    snippet = text

                            clean_body = f"# {title}\n\n{snippet}" if title else snippet
                            if clean_body:
                                sha = hashlib.sha256(clean_body.encode("utf-8")).hexdigest()
                                docs.append(
                                    ScrapedDocumentSchema(
                                        url=canon_url,
                                        raw_html=clean_body,
                                        clean_markdown=clean_body,
                                        sha256_hash=sha,
                                    )
                                )
                        if docs:
                            return docs
                    except Exception as parse_err:
                        logger.debug(f"XML parse attempt failed: {parse_err}")

                    # 2. Fallback to JSON parsing if API returns JSON
                    try:
                        data = resp.json()
                        nlm_list = (
                            data.get("nlmSearchResult", {})
                            .get("list", {})
                            .get("document", [])
                        )
                        if isinstance(nlm_list, dict):
                            nlm_list = [nlm_list]

                        for doc in nlm_list[:max_results]:
                            url = doc.get("@url", f"{MEDLINEPLUS_WEB_BASE}/healthtopics.html")
                            canon_url = canonicalize_url(url) or url
                            title = ""
                            snippet = ""
                            for content in doc.get("content", []):
                                name = content.get("@name", "")
                                val = content.get("$", "")
                                if name == "title":
                                    title = val
                                elif name in ("FullSummary", "snippet"):
                                    snippet = val

                            clean_body = f"# {title}\n\n{snippet}" if title else snippet
                            if clean_body:
                                sha = hashlib.sha256(clean_body.encode("utf-8")).hexdigest()
                                docs.append(
                                    ScrapedDocumentSchema(
                                        url=canon_url,
                                        raw_html=clean_body,
                                        clean_markdown=clean_body,
                                        sha256_hash=sha,
                                    )
                                )
                        if docs:
                            return docs
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug(f"MedlinePlus live HTTP query bypassed ({exc}); engaging local FTS5 store.")

        return []

    def _fetch_local_fallback(self, query: str, max_results: int) -> list[ScrapedDocumentSchema]:
        """Query local NIH MedlinePlus & MedQuAD knowledge base via SQLite FTS5."""
        if not self.db_path.exists():
            return []

        docs: list[ScrapedDocumentSchema] = []
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Sanitize query for FTS5 (remove punctuation that breaks FTS syntax)
            sanitized_terms = [re.sub(r"[^\w\s]", "", t).strip() for t in query.split()]
            sanitized_terms = [t for t in sanitized_terms if t]
            if not sanitized_terms:
                sanitized_terms = [query]

            fts_query = " OR ".join(sanitized_terms[:4])

            # Query clinical_guidelines_fts
            cursor.execute(
                """
                SELECT topic, cui, question, answer, source_institute 
                FROM clinical_guidelines 
                WHERE id IN (
                    SELECT rowid FROM clinical_guidelines_fts 
                    WHERE clinical_guidelines_fts MATCH ? 
                    ORDER BY rank 
                    LIMIT ?
                )
                """,
                (fts_query, max_results),
            )
            rows = cursor.fetchall()

            # Fallback to LIKE if FTS yielded 0 rows
            if not rows:
                cursor.execute(
                    """
                    SELECT topic, cui, question, answer, source_institute 
                    FROM clinical_guidelines 
                    WHERE topic LIKE ? OR question LIKE ? OR answer LIKE ? 
                    LIMIT ?
                    """,
                    (f"%{sanitized_terms[0]}%", f"%{sanitized_terms[0]}%", f"%{sanitized_terms[0]}%", max_results),
                )
                rows = cursor.fetchall()

            conn.close()

            for row in rows:
                topic, cui, question, answer, source = row
                markdown = (
                    f"### {topic} - NIH / MedlinePlus Guidance\n\n"
                    f"**Clinical Question**: {question}\n\n"
                    f"**Medical Summary**: {answer}\n\n"
                    f"*Source: {source}*"
                )
                topic_slug = quote_plus(topic.lower().replace(" ", "_"))
                url = f"{MEDLINEPLUS_WEB_BASE}/topic/{topic_slug}"
                sha = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

                docs.append(
                    ScrapedDocumentSchema(
                        url=url,
                        raw_html=markdown,
                        clean_markdown=markdown,
                        sha256_hash=sha,
                    )
                )
        except Exception as e:
            logger.warning(f"Error querying local MedlinePlus clinical evidence: {e}")

        return docs

    def get_disease_evidence(
        self, disease_name: str, present_symptoms: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Retrieves structured evidence: Diagnostic Odds Ratios + NIH clinical synopsis."""
        evidence_summary: dict[str, Any] = {
            "disease": disease_name,
            "source": "NIH / NLM MedlinePlus",
            "url": f"{MEDLINEPLUS_WEB_BASE}/topic/{quote_plus(disease_name.lower())}",
            "overview": "",
            "diagnostic_odds_ratios": [],
            "pathognomonic_hallmarks": [],
            "when_to_seek_er": False,
        }

        if not self.db_path.exists():
            return evidence_summary

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # 1. Fetch text overview from clinical_guidelines (try exact, then keyword)
            cursor.execute(
                "SELECT answer FROM clinical_guidelines WHERE topic = ? LIMIT 1",
                (disease_name,),
            )
            guideline_row = cursor.fetchone()
            if not guideline_row:
                cursor.execute(
                    "SELECT answer FROM clinical_guidelines WHERE topic LIKE ? LIMIT 1",
                    (f"%{disease_name}%",),
                )
                guideline_row = cursor.fetchone()

            # If still not found, search individual significant keywords (e.g. 'Asthma' from 'Bronchial Asthma')
            keywords = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", disease_name)]
            if not guideline_row and keywords:
                for kw in keywords:
                    cursor.execute(
                        "SELECT answer FROM clinical_guidelines WHERE topic LIKE ? LIMIT 1",
                        (f"%{kw}%",),
                    )
                    guideline_row = cursor.fetchone()
                    if guideline_row:
                        break

            if guideline_row:
                evidence_summary["overview"] = guideline_row[0]

            # 2. Fetch quantitative DOR and pathognomonic symptoms
            cursor.execute(
                """
                SELECT symptom, p_s_given_disease, odds_ratio, is_pathognomonic 
                FROM evidence_odds 
                WHERE disease = ? 
                ORDER BY odds_ratio DESC 
                LIMIT 10
                """,
                (disease_name,),
            )
            odds_rows = cursor.fetchall()

            # Fallback search in evidence_odds by keyword
            if not odds_rows and keywords:
                for kw in keywords:
                    cursor.execute(
                        """
                        SELECT symptom, p_s_given_disease, odds_ratio, is_pathognomonic 
                        FROM evidence_odds 
                        WHERE disease LIKE ? 
                        ORDER BY odds_ratio DESC 
                        LIMIT 10
                        """,
                        (f"%{kw}%",),
                    )
                    odds_rows = cursor.fetchall()
                    if odds_rows:
                        break
            for r in odds_rows:
                sym, p_s, dor, pathog = r
                if pathog:
                    evidence_summary["pathognomonic_hallmarks"].append(sym)
                evidence_summary["diagnostic_odds_ratios"].append(
                    {
                        "symptom": sym,
                        "probability": float(p_s),
                        "diagnostic_odds_ratio": float(dor),
                        "is_hallmark": bool(pathog),
                    }
                )

            # Check for high-acuity hallmarks
            high_acuity_keywords = [
                "heart attack",
                "stroke",
                "severe chest pain",
                "difficulty breathing",
                "anaphylaxis",
                "sepsis",
            ]
            if any(k in disease_name.lower() or k in evidence_summary["overview"].lower() for k in high_acuity_keywords):
                evidence_summary["when_to_seek_er"] = True

            conn.close()
        except Exception as e:
            logger.warning(f"Error fetching structured disease evidence for {disease_name}: {e}")

        return evidence_summary
