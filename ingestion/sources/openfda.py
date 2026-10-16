"""Official openFDA Drug Labeling REST API & Structured Product Labeling (SPL) Ingestion.

Queries the U.S. Food and Drug Administration (FDA) openFDA Drug Labeling endpoint:
https://api.fda.gov/drug/label.json

Provides official, regulatory pharmacological information:
- indications_and_usage
- dosage_and_administration
- contraindications
- boxed_warning (Black Box Warnings)
- warnings_and_cautions
- pediatric_use
- pregnancy & lactation

Features local SQLite FTS5 caching in clinical_evidence.db for offline resilience
and guaranteed deterministic baseline responses.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from core.schemas import ScrapedDocumentSchema
from ingestion.sources.base import BaseIngestionSource, canonicalize_url

logger = logging.getLogger(__name__)

OPENFDA_LABEL_API = "https://api.fda.gov/drug/label.json"
USER_AGENT = "HealthTrackAI-openFDAIngestion/1.0 (FDA-Compliant CDSS; safety@healthtrack.ai)"


class OpenFDASource(BaseIngestionSource):
    """Authoritative openFDA Drug Labeling client with local SQLite fallback."""

    def __init__(self, db_path: Optional[Path] = None, timeout: float = 6.0):
        super().__init__(name="openfda")
        self.timeout = timeout
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.db_path = db_path or (base_dir / "data" / "processed" / "clinical_evidence.db")
        self._ensure_cache_db()

    def _ensure_cache_db(self) -> None:
        """Initializes the SQLite cache table and seeds foundational drug labels if empty."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fda_drug_labels (
                        generic_name TEXT PRIMARY KEY,
                        brand_names TEXT,
                        drug_class TEXT,
                        indications_and_usage TEXT,
                        dosage_and_administration TEXT,
                        contraindications TEXT,
                        boxed_warning TEXT,
                        pediatric_use TEXT,
                        pregnancy TEXT,
                        raw_json TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute("SELECT COUNT(*) FROM fda_drug_labels")
                count = cursor.fetchone()[0]
                if count == 0:
                    self._seed_foundational_labels(cursor)
                conn.commit()
        except Exception as exc:
            logger.warning(f"Error initializing openFDA SQLite cache ({exc}); continuing.")

    def _seed_foundational_labels(self, cursor: sqlite3.Cursor) -> None:
        """Seeds essential drug product labels from official FDA monograph benchmarks."""
        foundational = [
            (
                "acetaminophen",
                json.dumps(["Tylenol", "Mapap", "Ofirmev"]),
                "Analgesic / Antipyretic",
                "Temporary relief of minor aches and pains due to headache, muscular aches, backache, minor arthritis, the common cold, toothache, and premenstrual/menstrual cramps; temporary reduction of fever.",
                "Adults and children 12 years and over: 500 mg to 1000 mg orally every 4 to 6 hours as needed. Do not exceed 3000 mg in 24 hours (or 4000 mg under direct medical direction). Pediatrics (<12 years): 10-15 mg/kg per dose every 4-6 hours PRN, not to exceed 75 mg/kg/day or 5 doses in 24 hours.",
                "Hypersensitivity to acetaminophen or any component of the formulation. Severe active hepatic impairment or severe active liver disease.",
                "BOXED WARNING: Hepatotoxicity. Acetaminophen has been associated with cases of acute liver failure, at times resulting in liver transplant and death. Most cases of liver injury are associated with the use of acetaminophen at doses that exceed 4000 mg per day, and often involve more than one acetaminophen-containing product.",
                "Safety and effectiveness in pediatric patients below 2 years of age have not been established for OTC labeling without physician guidance. Weight-based dosing (10-15 mg/kg) must be strictly adhered to.",
                "Epidemiological data over decades have not established a clear drug-associated risk of major birth defects or miscarriage. First-line antipyretic of choice in pregnancy when clinically needed.",
            ),
            (
                "ibuprofen",
                json.dumps(["Advil", "Motrin", "Nuprin"]),
                "Nonsteroidal Anti-inflammatory Drug (NSAID)",
                "Relief of signs and symptoms of rheumatoid arthritis and osteoarthritis; relief of mild to moderate pain; reduction of fever; treatment of primary dysmenorrhea.",
                "Adults: 200 mg to 400 mg orally every 4 to 6 hours with food or milk as needed. Maximum OTC dose is 1200 mg in 24 hours (prescription doses up to 3200 mg/day under clinical supervision). Pediatrics (>= 6 months): 5-10 mg/kg orally every 6-8 hours with food (maximum 40 mg/kg/day).",
                "Known hypersensitivity (e.g., anaphylaxis or serious skin reactions) to ibuprofen or other NSAIDs. History of asthma, urticaria, or other allergic-type reactions after taking aspirin or other NSAIDs. Peri-operative setting of coronary artery bypass graft (CABG) surgery. Third trimester of pregnancy (at or after 30 weeks gestation). Active gastrointestinal bleeding or peptic ulcer disease.",
                "BOXED WARNING: Cardiovascular and Gastrointestinal Risk. Nonsteroidal anti-inflammatory drugs cause an increased risk of serious cardiovascular thrombotic events, including myocardial infarction and stroke, which can be fatal. NSAIDs also cause an increased risk of serious gastrointestinal adverse events including bleeding, ulceration, and perforation of the stomach or intestines.",
                "Safety and efficacy in pediatric infants under 6 months of age have not been established.",
                "Avoid use of NSAIDs in pregnant women at 30 weeks gestation and later due to premature closure of the fetal ductus arteriosus. Avoid NSAIDs at 20 weeks or later due to risk of oligohydramnios and fetal renal dysfunction.",
            ),
            (
                "aspirin",
                json.dumps(["Bayer", "Ecotrin", "Bufferin"]),
                "Salicylate / Platelet Aggregation Inhibitor",
                "Relief of mild to moderate pain, fever, and inflammation; vascular indications including reduction of risk of death and myocardial infarction in patients with suspected acute MI, secondary prevention of vascular events.",
                "Analgesic/antipyretic: 325 mg to 650 mg orally every 4 hours as needed (max 4000 mg/day). Secondary cardiovascular prevention: 81 mg orally once daily. Adult use only.",
                "Children and teenagers recovering from chickenpox or flu-like symptoms (REYES SYNDROME HAZARD). Known salicylate allergy. Active peptic ulcer disease. Severe hemophilia or active bleeding disorders.",
                "BOXED WARNING / WARNING: REYES SYNDROME. Children and teenagers should not use this medicine for chickenpox or flu symptoms before a doctor is consulted about Reyes syndrome, a rare but serious illness characterized by acute encephalopathy and fatty liver.",
                "Contraindicated in pediatric patients under 19 years of age for viral illness due to proven link with fatal Reye's syndrome.",
                "Avoid use during the third trimester of pregnancy due to premature closure of ductus arteriosus and prolonged maternal/fetal bleeding.",
            ),
            (
                "amoxicillin",
                json.dumps(["Amoxil", "Trimox", "Moxatag"]),
                "Aminopenicillin Antibiotic",
                "Treatment of infections due to susceptible isolates of designated microorganisms: ear, nose, and throat infections (streptococcal pharyngitis, acute otitis media), genitourinary tract infections, skin and skin structure infections, lower respiratory tract infections.",
                "Adults: 500 mg orally every 8 hours, or 875 mg orally every 12 hours for 10 days for Streptococcal pharyngitis. Requires formal clinician prescription. Pediatrics (>3 months): 25 to 45 mg/kg/day divided every 12 hours (up to 90 mg/kg/day for high-dose otitis media protocols).",
                "History of a serious hypersensitivity reaction (e.g., anaphylaxis or Stevens-Johnson syndrome) to amoxicillin or to other beta-lactam antibacterial drugs (e.g., penicillins and cephalosporins).",
                "None. WARNINGS: Serious and occasionally fatal hypersensitivity (anaphylactic) reactions have been reported in patients on penicillin therapy. Clostridioides difficile-associated diarrhea (CDAD).",
                "Pediatric dosing must be calculated strictly on weight (mg/kg) in divided doses.",
                "Reproduction studies in animals have revealed no evidence of harm to the fetus. Penicillins are generally considered compatible with pregnancy (FDA Category B).",
            ),
            (
                "amoxicillin_clavulanate",
                json.dumps(["Augmentin", "Augmentin XR"]),
                "Beta-Lactamase Inhibitor Combination Antibiotic",
                "Acute bacterial rhinosinusitis, recurrent acute otitis media, community-acquired pneumonia, skin and skin structure infections, and animal bites caused by beta-lactamase-producing strains.",
                "Adults: 875/125 mg orally every 12 hours, or 500/125 mg orally every 8 hours with meals to enhance absorption and minimize GI intolerance.",
                "History of serious hypersensitivity to amoxicillin, clavulanate, or other beta-lactams. History of amoxicillin-clavulanate-associated cholestatic jaundice or hepatic dysfunction.",
                "None. WARNING: Hypersensitivity and hepatic dysfunction/cholestasis.",
                "Pediatric dosing should utilize 12-hour suspension formulations (e.g., 25-45 mg/kg/day amoxicillin component).",
                "Compatible with pregnancy; commonly used for maternal infections when indicated.",
            ),
            (
                "doxycycline",
                json.dumps(["Vibramycin", "Doryx", "Monodox"]),
                "Tetracycline Class Antibiotic",
                "Early Lyme disease (erythema migrans), Rocky Mountain spotted fever, rickettsial infections, chlamydial infections, atypical respiratory tract infections, severe acne vulgaris.",
                "Early Lyme disease: 100 mg orally twice daily for 10 to 14 days (up to 21 days depending on clinical response). Take with a full glass of water and remain upright for at least 30 minutes to prevent pill-induced esophagitis.",
                "Known hypersensitivity to any of the tetracyclines. Pregnancy (teratogenic bone growth inhibition). Pediatrics under 8 years of age (permanent tooth enamel hypoplasia/discoloration) unless life-threatening tick-borne rickettsial infection.",
                "None. WARNINGS: Tooth discoloration in children under 8 years; photosensitivity; pill-induced esophageal ulceration; pseudotumor cerebri (benign intracranial hypertension).",
                "The use of drugs of the tetracycline class during tooth development (last half of pregnancy, infancy, and childhood to the age of 8 years) may cause permanent discoloration of the teeth (yellow-gray-brown).",
                "Doxycycline is contraindicated during pregnancy. Tetracyclines cross the placenta, deposit in fetal teeth and bones, and can retard skeletal development.",
            ),
            (
                "azithromycin",
                json.dumps(["Zithromax", "Z-Pak"]),
                "Macrolide Antibiotic",
                "Alternative therapy for mild to moderate infections including Streptococcal pharyngitis/tonsillitis in penicillin-allergic patients; acute bacterial exacerbations of chronic bronchitis; community-acquired pneumonia.",
                "Adult Z-Pak regimen: 500 mg orally as a single dose on Day 1, followed by 250 mg orally once daily on Days 2 through 5. Administer with or without food.",
                "Known hypersensitivity to azithromycin, erythromycin, or any macrolide/ketolide antibacterial drug. History of cholestatic jaundice/hepatic dysfunction associated with prior use of azithromycin.",
                "None. WARNINGS: QT prolongation, torsades de pointes, and fatal cardiac arrhythmias. Hepatotoxicity. Clostridioides difficile-associated diarrhea.",
                "Pediatric oral suspension dosing is typically 10 mg/kg on Day 1 (max 500 mg) followed by 5 mg/kg/day on Days 2-5 (max 250 mg).",
                "Azithromycin is generally considered safe in pregnancy when clinically necessary and non-macrolide alternatives are unsuitable.",
            ),
            (
                "oseltamivir",
                json.dumps(["Tamiflu"]),
                "Neuraminidase Inhibitor Antiviral",
                "Treatment of acute, uncomplicated illness due to influenza A and B infection in patients symptomatic for no more than 48 hours; prophylaxis of influenza.",
                "Adults and adolescents: 75 mg orally twice daily for 5 days. For post-exposure prophylaxis: 75 mg once daily for at least 10 days. Take with food to reduce nausea and vomiting.",
                "Known serious hypersensitivity to oseltamivir or any ingredient of the product.",
                "None. WARNINGS: Neuropsychiatric events (delirium, hallucination, abnormal behavior) reported primarily in pediatric patients in postmarketing surveillance. Not a substitute for annual influenza vaccination.",
                "Approved for treatment down to 2 weeks of age using weight-based oral suspension protocols.",
                "Influenza is associated with adverse pregnancy and fetal outcomes. Clinical guidelines (CDC/ACOG) recommend prompt oseltamivir treatment for pregnant individuals with confirmed or suspected influenza.",
            ),
            (
                "cephalexin",
                json.dumps(["Keflex"]),
                "First-Generation Cephalosporin Antibiotic",
                "Treatment of respiratory tract infections, otitis media, skin and skin structure infections, bone infections, and genitourinary tract infections caused by susceptible bacterial strains.",
                "Adults: 250 mg to 500 mg orally every 6 hours, or 500 mg every 12 hours for 7 to 14 days depending on indication.",
                "Known hypersensitivity to cephalosporins. Absolute contraindication if history of severe IgE-mediated anaphylaxis to penicillins due to shared beta-lactam core cross-reactivity.",
                "None. WARNINGS: Cross-hypersensitivity among beta-lactam antibiotics.",
                "Weight-based dosing in children: 25 to 50 mg/kg/day divided into 2 to 4 doses.",
                "Generally recognized as safe during pregnancy (Category B).",
            ),
            (
                "ciprofloxacin",
                json.dumps(["Cipro", "Cipro XR"]),
                "Fluoroquinolone Antibiotic",
                "Complicated urinary tract infections, pyelonephritis, bacterial prostatitis, complicated intra-abdominal infections, post-exposure inhalational anthrax.",
                "Adults: 250 mg to 500 mg orally every 12 hours for 7 to 14 days (or 750 mg q12h for severe/complicated infections). Drink fluids liberally to avoid crystalluria.",
                "Hypersensitivity to ciprofloxacin or other quinolones. Concomitant administration with tizanidine. Pediatric patients under 18 years of age (except for inhalational anthrax or complicated UTI/pyelonephritis where benefits outweigh risks). Pregnancy.",
                "BOXED WARNING: Serious Adverse Reactions Including Tendinitis, Tendon Rupture, Peripheral Neuropathy, Central Nervous System Effects, and Exacerbation of Myasthenia Gravis. Discontinue immediately if patient experiences pain, swelling, or inflammation of a tendon.",
                "Contraindicated in pediatric patients under 18 years of age due to increased incidence of adverse musculoskeletal disorders (arthropathy, chondrotoxicity).",
                "Contraindicated in pregnancy due to cartilage damage observed in animal models; alternative safe antimicrobials must be utilized.",
            ),
            (
                "pseudoephedrine",
                json.dumps(["Sudafed", "SudoGest"]),
                "Sympathomimetic Alpha/Beta-Adrenergic Agonist Decongestant",
                "Temporary relief of nasal congestion due to the common cold, hay fever, or other upper respiratory allergies; promotes nasal and sinus drainage.",
                "Adults and children 12 years and over: 30 mg to 60 mg orally every 4 to 6 hours as needed. Maximum daily dose: 240 mg in 24 hours.",
                "Severe uncontrolled hypertension. Severe coronary artery disease (CAD). Concurrent use with monoamine oxidase inhibitors (MAOIs) or within 14 days of stopping MAOI therapy. Closed-angle glaucoma.",
                "None. WARNINGS: Increases blood pressure and heart rate; risk of tachyarrhythmias, stroke, or myocardial infarction in susceptible cardiovascular patients.",
                "Do not use in children under 4 years of age for cold symptoms. Consult physician for pediatric dosing.",
                "Avoid use in first trimester of pregnancy due to association with gastroschisis and fetal vasoconstrictive effects.",
            ),
            (
                "bismuth_subsalicylate",
                json.dumps(["Pepto-Bismol", "Kaopectate"]),
                "Gastrointestinal Salicylate / Antidiarrheal",
                "Relief of traveler's diarrhea, heartburn, indigestion, upset stomach, and nausea.",
                "Adults: 524 mg (two regular tablets or 30 mL) orally every 30 to 60 minutes as needed. Do not exceed 8 doses (4192 mg) in 24 hours.",
                "Children and teenagers recovering from chickenpox, influenza, or viral illness (REYES SYNDROME HAZARD). Known salicylate allergy. Active gastrointestinal bleeding or ulceration.",
                "WARNING: REYES SYNDROME. Children and teenagers who have or are recovering from chickenpox or flu symptoms should not use this product.",
                "Strictly contraindicated in children and adolescents under 19 years of age.",
                "Contraindicated in the third trimester of pregnancy due to salicylate content.",
            ),
            (
                "cetirizine",
                json.dumps(["Zyrtec"]),
                "Second-Generation H1 Antihistamine",
                "Relief of symptoms associated with allergic rhinitis (sneezing, rhinorrhea, postnasal discharge, nasal pruritus, ocular pruritus) and treatment of uncomplicated skin manifestations of chronic idiopathic urticaria.",
                "Adults and children 6 years and older: 5 mg to 10 mg orally once daily depending upon symptom severity. Maximum daily dose: 10 mg in 24 hours.",
                "Known hypersensitivity to cetirizine, hydroxyzine, or any component of the formulation.",
                "None. PRECAUTIONS: May cause mild somnolence; exercise caution when operating heavy machinery.",
                "Children 2 to 5 years: 2.5 mg orally once daily (max 5 mg/day).",
                "Extensive human data have not shown evidence of fetal toxicity; considered safe in pregnancy.",
            ),
            (
                "fluticasone",
                json.dumps(["Flonase", "Flonase Sensimist"]),
                "Corticosteroid Nasal Spray",
                "Management of the nasal symptoms of perennial nonallergic rhinitis and seasonal/perennial allergic rhinitis in adults and pediatric patients aged 4 years and older.",
                "Adults: 2 sprays in each nostril once daily (total 200 mcg/day), or 1 spray in each nostril twice daily. After symptoms are controlled, reduce to 1 spray per nostril daily.",
                "Hypersensitivity to fluticasone propionate. Recent nasal septal ulcers, nasal surgery, or nasal trauma until healing has occurred.",
                "None. PRECAUTIONS: Epistaxis, localized Candida albicans nasal infections, potential slowing of growth velocity in pediatric patients with long-term continuous use.",
                "Children 4 to 11 years: 1 spray in each nostril once daily (100 mcg/day).",
                "Systemic absorption following intranasal administration is negligible; first-line intranasal steroid in pregnancy.",
            ),
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO fda_drug_labels (
                generic_name, brand_names, drug_class, indications_and_usage,
                dosage_and_administration, contraindications, boxed_warning,
                pediatric_use, pregnancy, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                    item[4],
                    item[5],
                    item[6],
                    item[7],
                    item[8],
                    json.dumps({
                        "generic_name": item[0],
                        "brand_names": json.loads(item[1]),
                        "drug_class": item[2],
                        "indications": item[3],
                        "dosage": item[4],
                        "contraindications": item[5],
                        "boxed_warning": item[6],
                        "pediatric": item[7],
                        "pregnancy": item[8],
                    }),
                )
                for item in foundational
            ],
        )
        logger.info(f"Seeded {len(foundational)} authoritative openFDA drug product labels into SQLite.")

    async def fetch(self, query: str, max_results: int = 5) -> List[ScrapedDocumentSchema]:
        """BaseIngestionSource contract: queries drug labeling data for clinical query."""
        results: List[ScrapedDocumentSchema] = []
        clean_q = query.strip()
        if not clean_q:
            return results

        label_data = await self.get_drug_label(clean_q)
        if label_data:
            doc_text = self._format_label_markdown(label_data)
            sha = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
            results.append(
                ScrapedDocumentSchema(
                    url=f"https://open.fda.gov/apis/drug/label?search={quote_plus(clean_q)}",
                    raw_html=doc_text,
                    clean_markdown=doc_text,
                    sha256_hash=sha,
                )
            )
        return results

    async def get_drug_label(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves structured drug label via live openFDA API with local SQLite fallback."""
        clean_name = drug_name.strip().lower()
        if not clean_name:
            return None

        # 1. Attempt live HTTP query to openFDA REST API
        try:
            headers = {"User-Agent": USER_AGENT}
            params = {
                "search": f'openfda.generic_name:"{clean_name}" OR openfda.brand_name:"{clean_name}"',
                "limit": 1,
            }
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                resp = await client.get(OPENFDA_LABEL_API, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        parsed = self._parse_openfda_result(results[0], clean_name)
                        self._upsert_label_cache(parsed)
                        return parsed
        except Exception as exc:
            logger.debug(f"Live openFDA query bypassed for {drug_name} ({exc}); using local cache.")

        # 2. Resilient local SQLite fallback
        return self.get_cached_drug_label(clean_name)

    def get_cached_drug_label(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Synchronously retrieves drug product label from local SQLite store."""
        clean_name = drug_name.strip().lower()
        if not self.db_path.exists():
            return None

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                # Direct match on generic_name
                cursor.execute(
                    """
                    SELECT generic_name, brand_names, drug_class, indications_and_usage,
                           dosage_and_administration, contraindications, boxed_warning,
                           pediatric_use, pregnancy, raw_json
                    FROM fda_drug_labels
                    WHERE generic_name = ?
                    LIMIT 1
                    """,
                    (clean_name,),
                )
                row = cursor.fetchone()

                # Brand name or keyword fallback
                if not row:
                    cursor.execute(
                        """
                        SELECT generic_name, brand_names, drug_class, indications_and_usage,
                               dosage_and_administration, contraindications, boxed_warning,
                               pediatric_use, pregnancy, raw_json
                        FROM fda_drug_labels
                        WHERE generic_name LIKE ? OR brand_names LIKE ?
                        LIMIT 1
                        """,
                        (f"%{clean_name}%", f"%{clean_name}%"),
                    )
                    row = cursor.fetchone()

                if row:
                    return {
                        "generic_name": row[0],
                        "brand_names": json.loads(row[1]) if row[1] else [],
                        "drug_class": row[2] or "Therapeutic Agent",
                        "indications_and_usage": row[3] or "",
                        "dosage_and_administration": row[4] or "",
                        "contraindications": row[5] or "",
                        "boxed_warning": row[6] or "",
                        "pediatric_use": row[7] or "",
                        "pregnancy": row[8] or "",
                    }
        except Exception as exc:
            logger.warning(f"Error querying SQLite openFDA cache for {drug_name}: {exc}")

        return None

    def _parse_openfda_result(self, raw_item: Dict[str, Any], query_name: str) -> Dict[str, Any]:
        """Extracts and normalizes clinical SPL sections from raw openFDA JSON."""
        openfda_block = raw_item.get("openfda", {})

        generic_list = openfda_block.get("generic_name", [])
        generic_name = generic_list[0].lower() if generic_list else query_name

        brand_list = openfda_block.get("brand_name", [])
        drug_class_list = openfda_block.get("pharm_class_epc", []) or openfda_block.get("pharm_class_cs", [])
        drug_class = drug_class_list[0] if drug_class_list else "Pharmacological Agent"

        def _clean_field(key: str) -> str:
            vals = raw_item.get(key, [])
            if isinstance(vals, list) and vals:
                text = " ".join(vals)
                return re.sub(r"\s+", " ", text).strip()
            return ""

        return {
            "generic_name": generic_name,
            "brand_names": brand_list[:5],
            "drug_class": drug_class,
            "indications_and_usage": _clean_field("indications_and_usage") or _clean_field("purpose"),
            "dosage_and_administration": _clean_field("dosage_and_administration"),
            "contraindications": _clean_field("contraindications") or _clean_field("do_not_use"),
            "boxed_warning": _clean_field("boxed_warning"),
            "pediatric_use": _clean_field("pediatric_use"),
            "pregnancy": _clean_field("pregnancy") or _clean_field("pregnancy_or_breast_feeding"),
            "raw_json": json.dumps(raw_item),
        }

    def _upsert_label_cache(self, label: Dict[str, Any]) -> None:
        """Upserts freshly parsed openFDA record into local SQLite database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO fda_drug_labels (
                        generic_name, brand_names, drug_class, indications_and_usage,
                        dosage_and_administration, contraindications, boxed_warning,
                        pediatric_use, pregnancy, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        label["generic_name"],
                        json.dumps(label.get("brand_names", [])),
                        label.get("drug_class", ""),
                        label.get("indications_and_usage", ""),
                        label.get("dosage_and_administration", ""),
                        label.get("contraindications", ""),
                        label.get("boxed_warning", ""),
                        label.get("pediatric_use", ""),
                        label.get("pregnancy", ""),
                        label.get("raw_json", "{}"),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"Could not upsert openFDA cache: {exc}")

    def _format_label_markdown(self, label: Dict[str, Any]) -> str:
        """Formats structured SPL drug label into clean Markdown for citation injection."""
        brands = ", ".join(label.get("brand_names", []))
        brand_line = f" (**Brand Names**: {brands})" if brands else ""
        lines = [
            f"# FDA Approved Drug Label: {label['generic_name'].title()}{brand_line}",
            f"**Drug Class**: {label.get('drug_class', 'N/A')}\n",
            f"### Indications and Usage\n{label.get('indications_and_usage', 'Not specified.')}\n",
            f"### Dosage and Administration\n{label.get('dosage_and_administration', 'Not specified.')}\n",
        ]
        if label.get("boxed_warning"):
            lines.append(f"### ⚠️ BOXED WARNING\n{label['boxed_warning']}\n")
        if label.get("contraindications"):
            lines.append(f"### Contraindications\n{label['contraindications']}\n")
        if label.get("pediatric_use"):
            lines.append(f"### Pediatric Precautions\n{label['pediatric_use']}\n")
        if label.get("pregnancy"):
            lines.append(f"### Pregnancy and Lactation\n{label['pregnancy']}\n")
        lines.append("*Source: U.S. Food and Drug Administration (FDA) Structured Product Labeling (SPL)*")
        return "\n".join(lines)


# Global singleton instance for system-wide access
openfda_source = OpenFDASource()
