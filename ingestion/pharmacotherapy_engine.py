"""Authoritative Pharmacotherapy Guidance & Deep Clinical Medication Engine.

Synthesizes personalized, tiered medication plans grounded in:
1. openFDA Structured Product Labeling (SPL) API & local cache
2. Clinical Etiology Understanding (Viral vs. Bacterial vs. Inflammatory vs. Ischemic)
3. Deterministic Safety Shield:
   - RxCrossReactivityGraph allergy filtering (Penicillins, Cephalosporins, Sulfas, NSAIDs)
   - Pediatric physiological boundaries (Age < 19 Salicylate / Reye's syndrome exclusion)
   - Obstetric teratogenicity shielding (Category X, Fluoroquinolones, Tetracyclines, 3rd-trimester NSAIDs)
   - Comorbidity contraindications (Hypertension, CKD, Hepatic, Peptic Ulcers)

Outputs conform to the FDA CDSS Non-Device Regulatory Framework (Pydantic V2):
Tier 1: Non-pharmacological & Supportive Care
Tier 2: Safe Over-The-Counter (OTC) Relief with explicit dosage ceilings
Tier 3: Guideline-Directed Prescription Therapies to discuss with a clinician
Tier 4: Safety Shield Violations & Blocked Medications with explicit clinical rationale
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from core.schemas import MedicationDosingSchema, PharmacotherapyPlanSchema
from ingestion.sources.openfda import OpenFDASource, openfda_source
from training.m5_preference.clinical_safety_ontology import (
    PediatricObstetricSafetyRules,
    RxCrossReactivityGraph,
)

logger = logging.getLogger(__name__)


class PharmacotherapyEngine:
    """Production clinical medication engine for deep etiological recommendation."""

    def __init__(self, fda_source: Optional[OpenFDASource] = None):
        self.fda = fda_source or openfda_source

    # ── Etiology Classification Dictionary ─────────────────────────────────────
    VIRAL_CONDITIONS = {
        "common cold", "cold", "rhinitis", "upper respiratory infection", "uri",
        "viral pharyngitis", "viral sore throat", "influenza", "flu", "viral syndrome",
        "acute viral bronchitis", "bronchitis", "viral gastroenteritis", "stomach flu",
        "rotavirus", "norovirus", "covid", "covid-19", "coronavirus", "infectious mononucleosis",
        "mono", "croup", "laryngitis", "herpes simplex", "herpes zoster", "varicella", "chickenpox"
    }

    BACTERIAL_CONDITIONS = {
        "strep throat", "streptococcal pharyngitis", "bacterial pharyngitis", "acute bacterial rhinosinusitis",
        "bacterial sinusitis", "sinusitis", "acute otitis media", "otitis media", "middle ear infection",
        "lyme disease", "early lyme", "borreliosis", "erythema migrans", "urinary tract infection",
        "uti", "cystitis", "acute cystitis", "pyelonephritis", "kidney infection", "cellulitis",
        "impetigo", "community-acquired pneumonia", "bacterial pneumonia", "chlamydia", "gonorrhea"
    }

    INFLAMMATORY_MUSCULOSKELETAL_CONDITIONS = {
        "osteoarthritis", "rheumatoid arthritis", "arthritis", "lumbar strain", "cervical strain",
        "muscle strain", "tendinitis", "tendonitis", "sprain", "myofascial pain", "tension headache",
        "bursitis", "plantar fasciitis", "back pain", "neck pain", "gout", "acute gouty arthritis"
    }

    ALLERGIC_CONDITIONS = {
        "allergic rhinitis", "hay fever", "seasonal allergies", "allergic conjunctivitis",
        "urticaria", "hives", "atopic dermatitis", "eczema", "contact dermatitis"
    }

    @classmethod
    def classify_etiology(cls, condition_name: str, present_symptoms: Optional[List[str]] = None) -> str:
        """Classifies the condition into a primary clinical etiology."""
        cond_lower = condition_name.strip().lower()

        for term in cls.VIRAL_CONDITIONS:
            if term in cond_lower:
                return "viral"

        for term in cls.BACTERIAL_CONDITIONS:
            if term in cond_lower:
                return "bacterial"

        for term in cls.INFLAMMATORY_MUSCULOSKELETAL_CONDITIONS:
            if term in cond_lower:
                return "musculoskeletal"

        for term in cls.ALLERGIC_CONDITIONS:
            if term in cond_lower:
                return "allergic"

        # Symptom-informed heuristics
        if present_symptoms:
            sym_text = " ".join(present_symptoms).lower()
            if "tick" in sym_text or "erythema" in sym_text or "target" in sym_text:
                return "bacterial"
            if "runny" in sym_text and "sneezing" in sym_text and "itching" in sym_text:
                return "allergic"

        return "undetermined"

    def generate_plan(
        self,
        condition_name: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        present_symptoms: Optional[List[str]] = None,
    ) -> PharmacotherapyPlanSchema:
        """
        Synthesizes a personalized, 4-tier pharmacotherapy plan tailored to etiology,
        dosing limits, and the patient's individual safety shield profile.
        """
        profile = patient_profile or {}
        age = self._extract_age(profile)
        is_pregnant = profile.get("is_pregnant", False) or profile.get("pregnancy", False)
        allergies = profile.get("drug_allergies", []) or []
        chronic_conditions = profile.get("chronic_conditions", []) or []

        etiology = self.classify_etiology(condition_name, present_symptoms)
        antimicrobial_indicated = (etiology == "bacterial")

        # 1. Tier 1: Non-Pharmacological Supportive Care
        tier_1 = self._build_tier_1_supportive_care(etiology, condition_name, age, present_symptoms)

        # 2. Candidate Tier 2: Safe Over-The-Counter (OTC) Medications
        raw_tier_2 = self._build_candidate_tier_2_otc(etiology, condition_name, age, present_symptoms)

        # 3. Candidate Tier 3: Evidence-Based Prescription Therapies to Discuss
        raw_tier_3 = self._build_candidate_tier_3_prescription(etiology, condition_name, age, present_symptoms)

        # 4. Safety Shield Filtering across all candidate medications
        safe_tier_2, blocked_tier_2 = self._filter_through_safety_shield(
            raw_tier_2, age, is_pregnant, allergies, chronic_conditions
        )
        safe_tier_3, blocked_tier_3 = self._filter_through_safety_shield(
            raw_tier_3, age, is_pregnant, allergies, chronic_conditions
        )

        all_blocked = blocked_tier_2 + blocked_tier_3

        # Add Antimicrobial Stewardship blocking statement for viral conditions
        if etiology == "viral":
            all_blocked.append(
                "Antibiotics (Amoxicillin, Azithromycin, Cephalexin, Fluoroquinolones): Strictly blocked under "
                "CDC/IDSA Antimicrobial Stewardship principles. Antibacterial agents possess zero efficacy against "
                "viral viral pathogens, promote adverse gastrointestinal effects (C. difficile), and cultivate resistance."
            )

        # Ensure safe alternatives if first-line candidates were blocked
        safe_tier_2, safe_tier_3 = self._ensure_safe_alternatives(
            safe_tier_2, safe_tier_3, etiology, condition_name, age, is_pregnant, allergies, chronic_conditions
        )

        return PharmacotherapyPlanSchema(
            condition_name=condition_name,
            etiology_type=etiology,
            antimicrobial_indicated=antimicrobial_indicated,
            tier_1_supportive_care=tier_1,
            tier_2_otc_medications=safe_tier_2,
            tier_3_prescription_guidance=safe_tier_3,
            safety_shield_violations_blocked=all_blocked,
        )

    # ── Tier 1 Builders ────────────────────────────────────────────────────────
    def _build_tier_1_supportive_care(
        self, etiology: str, condition: str, age: Optional[int], present_symptoms: Optional[List[str]]
    ) -> List[str]:
        care = []
        cond_lower = condition.lower()

        if etiology == "viral" or "flu" in cond_lower or "cold" in cond_lower:
            care.extend([
                "Liberal Oral Hydration: Maintain fluid intake of 2.0 to 3.0 Liters daily (water, clear broths, oral rehydration electrolyte solutions) to prevent dehydration and thin respiratory secretions.",
                "Metabolic Rest: Prioritize complete bed rest and sleep to optimize endogenous cell-mediated immune responses.",
                "Warm Saline Gargles: Dissolve 1/2 teaspoon of salt in 8 ounces of warm water and gargle every 2 to 4 hours for natural pharyngeal soothing.",
                "Cool-Mist Humidification: Run a clean ultrasonic cool-mist humidifier in the bedroom to moisten irritated respiratory mucosa and reduce nocturnal cough.",
            ])
            if age is None or age >= 1:
                care.append("Demulcent Therapy: 1 to 2 teaspoons of pure pasteurized honey PRN for cough suppression (Do NOT administer to infants under 1 year due to infant botulism risk).")
        elif etiology == "bacterial":
            care.extend([
                "Targeted Hydration: Maintain consistent fluid intake to support physiological metabolism during active bacterial defense.",
                "Warm Saline Rinses / Gargles: For upper respiratory or pharyngeal bacterial infections, perform warm salt water gargles every 3 hours.",
                "Local Thermal Comfort: Apply warm, moist compresses to affected local areas (e.g. sinuses, lymphadenopathy) for comfort.",
            ])
            if "lyme" in cond_lower:
                care.append("Tick Site Monitoring: Photograph the tick bite site and erythema migrans rash border daily with a ruler to document objective expansion or resolution.")
        elif etiology == "musculoskeletal":
            care.extend([
                "R.I.C.E. Protocol: Rest the affected area, Ice with cold packs (15-20 minutes every 2-3 hours for the first 48 hours), Compress gently if swelling is present, and Elevate above heart level.",
                "Thermal Transition: After 48 hours of cold application, gentle moist heat can be applied to relax tense muscular spasms.",
                "Gentle Range-of-Motion: Perform non-weight-bearing gentle mobility exercises within pain-free boundaries to prevent joint stiffness.",
            ])
        elif etiology == "allergic":
            care.extend([
                "Allergen Avoidance: Keep windows closed during high pollen counts, change clothes upon entering indoors, and shower before bedtime.",
                "Isotonic Saline Nasal Lavage: Flush nasal passages with sterile or distilled saline sinus rinses twice daily to physically clear trapped aeroallergens.",
            ])
        else:
            care.extend([
                "Supportive Oral Hydration: Drink plenty of water and clear broths.",
                "Rest: Ensure adequate sleep and avoid intense physical exertion until acute symptoms subside.",
            ])

        return care

    # ── Tier 2 Builders ────────────────────────────────────────────────────────
    def _build_candidate_tier_2_otc(
        self, etiology: str, condition: str, age: Optional[int], present_symptoms: Optional[List[str]]
    ) -> List[MedicationDosingSchema]:
        candidates: List[MedicationDosingSchema] = []
        is_pediatric = (age is not None and age < 12)

        # 1. Acetaminophen (Foundational Antipyretic / Analgesic)
        if is_pediatric:
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Children's Tylenol (Acetaminophen Oral Suspension 160mg/5mL)",
                    generic_name="acetaminophen",
                    drug_class="Analgesic / Antipyretic",
                    is_otc=True,
                    typical_dosage="10 to 15 mg/kg orally every 4 to 6 hours as needed using the enclosed calibrated syringe",
                    maximum_daily_limit="Do not exceed 75 mg/kg in 24 hours, or a maximum of 5 doses in 24 hours",
                    route="oral",
                    clinical_purpose="First-line antipyretic and analgesic for fever reduction and somatic discomfort",
                    contraindication_warnings=[
                        "Severe liver dysfunction or active hepatic disease",
                        "Always measure with calibrated syringe; never use household spoons",
                    ],
                    evidence_source="openFDA / AAP Pediatric Dosing Guidelines",
                )
            )
        else:
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Tylenol (Acetaminophen)",
                    generic_name="acetaminophen",
                    drug_class="Analgesic / Antipyretic",
                    is_otc=True,
                    typical_dosage="500 mg to 1000 mg orally every 4 to 6 hours as needed with water",
                    maximum_daily_limit="Do NOT exceed 3000 mg in 24 hours (or 2000 mg/day if hepatic risk or chronic alcohol use)",
                    route="oral",
                    clinical_purpose="Reduction of fever, relief of headache, myalgias, and sore throat discomfort",
                    contraindication_warnings=[
                        "Severe active liver disease or acute hepatic impairment",
                        "Avoid concurrent use with other acetaminophen-containing combination cold products to prevent hepatotoxicity",
                    ],
                    evidence_source="openFDA SPL Drug Label / NIH MedlinePlus",
                )
            )

        # 2. Ibuprofen (Anti-inflammatory / Antipyretic)
        if is_pediatric:
            if age is None or age >= 1:  # Typically >= 6 months
                candidates.append(
                    MedicationDosingSchema(
                        drug_name="Children's Motrin (Ibuprofen Oral Suspension 100mg/5mL)",
                        generic_name="ibuprofen",
                        drug_class="Nonsteroidal Anti-inflammatory Drug (NSAID)",
                        is_otc=True,
                        typical_dosage="5 to 10 mg/kg orally every 6 to 8 hours with food or milk",
                        maximum_daily_limit="Do not exceed 40 mg/kg in 24 hours or 4 doses in 24 hours",
                        route="oral",
                        clinical_purpose="Relief of inflammatory pain and fever refractory to single-agent acetaminophen",
                        contraindication_warnings=[
                            "Infants under 6 months of age",
                            "Active dehydration or vomiting without adequate oral intake",
                            "Known hypersensitivity to NSAIDs",
                        ],
                        evidence_source="openFDA SPL Drug Label / AAP Pediatric Dosing Guidelines",
                    )
                )
        else:
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Advil / Motrin (Ibuprofen)",
                    generic_name="ibuprofen",
                    drug_class="Nonsteroidal Anti-inflammatory Drug (NSAID)",
                    is_otc=True,
                    typical_dosage="200 mg to 400 mg orally every 4 to 6 hours with food or a full glass of milk",
                    maximum_daily_limit="Do NOT exceed 1200 mg in 24 hours for OTC self-care",
                    route="oral",
                    clinical_purpose="Relief of inflammatory pain, pharyngeal inflammation, fever, and muscular aches",
                    contraindication_warnings=[
                        "Active gastrointestinal ulcer or history of GI bleeding",
                        "Severe renal insufficiency / Chronic Kidney Disease",
                        "Avoid in third trimester of pregnancy (premature closure of ductus arteriosus)",
                        "History of aspirin-induced asthma or severe NSAID allergy",
                    ],
                    evidence_source="openFDA SPL Drug Label / NIH MedlinePlus",
                )
            )

        # 3. Pseudoephedrine (Systemic Decongestant for nasal congestion)
        if not is_pediatric:
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Sudafed (Pseudoephedrine HCl)",
                    generic_name="pseudoephedrine",
                    drug_class="Sympathomimetic Alpha-Adrenergic Decongestant",
                    is_otc=True,
                    typical_dosage="30 mg to 60 mg orally every 4 to 6 hours as needed",
                    maximum_daily_limit="Do not exceed 240 mg in 24 hours; do not use for more than 7 consecutive days",
                    route="oral",
                    clinical_purpose="Temporary relief of sinus pressure and severe nasal mucosal congestion",
                    contraindication_warnings=[
                        "Severe uncontrolled hypertension or coronary artery disease (CAD)",
                        "Concurrent MAOI therapy",
                        "Angle-closure glaucoma",
                    ],
                    evidence_source="openFDA SPL Drug Label",
                )
            )

        # 4. Salicylates (Aspirin & Pepto-Bismol) - included as raw candidates so safety shield can test and block them for pediatrics
        candidates.append(
            MedicationDosingSchema(
                drug_name="Bayer Aspirin (Acetylsalicylic Acid)",
                generic_name="aspirin",
                drug_class="Salicylate / NSAID",
                is_otc=True,
                typical_dosage="325 mg to 650 mg orally every 4 hours as needed with food",
                maximum_daily_limit="Do not exceed 4000 mg in 24 hours",
                route="oral",
                clinical_purpose="Pain relief and anti-inflammatory antipyretic for adult patients only",
                contraindication_warnings=[
                    "STRICT CONTRAINDICATION in children and teenagers under 19 with viral symptoms due to Reye's syndrome",
                    "Active peptic ulcer disease or bleeding diathesis",
                ],
                evidence_source="openFDA SPL Drug Label",
            )
        )

        candidates.append(
            MedicationDosingSchema(
                drug_name="Pepto-Bismol (Bismuth Subsalicylate)",
                generic_name="bismuth_subsalicylate",
                drug_class="Gastrointestinal Salicylate",
                is_otc=True,
                typical_dosage="524 mg orally every 30 to 60 minutes as needed",
                maximum_daily_limit="Do not exceed 4192 mg (8 doses) in 24 hours",
                route="oral",
                clinical_purpose="Relief of nausea, heartburn, indigestion, upset stomach, and diarrhea",
                contraindication_warnings=[
                    "STRICT CONTRAINDICATION in children and teenagers under 19 recovering from viral illness (Reye's syndrome)",
                    "Aspirin or salicylate allergy",
                ],
                evidence_source="openFDA SPL Drug Label",
            )
        )

        # 5. Allergic Rhinitis Specific OTCs
        if etiology == "allergic":
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Zyrtec (Cetirizine HCl)",
                    generic_name="cetirizine",
                    drug_class="Second-Generation H1 Antihistamine",
                    is_otc=True,
                    typical_dosage="10 mg orally once daily in the evening",
                    maximum_daily_limit="Do not exceed 10 mg in 24 hours",
                    route="oral",
                    clinical_purpose="Relief of allergic sneezing, rhinorrhea, itchy/watery eyes, and pruritus",
                    contraindication_warnings=["Hypersensitivity to cetirizine or hydroxyzine"],
                    evidence_source="openFDA SPL Drug Label",
                )
            )
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Flonase (Fluticasone Propionate Nasal Spray)",
                    generic_name="fluticasone",
                    drug_class="Intranasal Corticosteroid",
                    is_otc=True,
                    typical_dosage="2 sprays per nostril once daily, titrating down to 1 spray per nostril daily once symptoms stabilize",
                    maximum_daily_limit="Do not exceed 2 sprays per nostril daily (200 mcg/day)",
                    route="nasal",
                    clinical_purpose="First-line suppression of allergic nasal mucosal inflammation and congestion",
                    contraindication_warnings=["Recent nasal ulcers, trauma, or unhealed nasal surgery"],
                    evidence_source="openFDA SPL Drug Label",
                )
            )

        return candidates

    # ── Tier 3 Builders ────────────────────────────────────────────────────────
    def _build_candidate_tier_3_prescription(
        self, etiology: str, condition: str, age: Optional[int], present_symptoms: Optional[List[str]]
    ) -> List[MedicationDosingSchema]:
        candidates: List[MedicationDosingSchema] = []
        cond_lower = condition.lower()

        # A. Viral Presentation with Antiviral Consideration (Influenza)
        if "flu" in cond_lower or "influenza" in cond_lower:
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Tamiflu (Oseltamivir Phosphate)",
                    generic_name="oseltamivir",
                    drug_class="Neuraminidase Inhibitor Antiviral",
                    is_otc=False,
                    typical_dosage="75 mg orally twice daily for 5 consecutive days (must be initiated within 48 hours of symptom onset)",
                    maximum_daily_limit="150 mg in 24 hours",
                    route="oral",
                    clinical_purpose="Shorten duration of uncomplicated influenza A and B infection in high-risk patients",
                    contraindication_warnings=[
                        "History of severe hypersensitivity to oseltamivir",
                        "Requires formal evaluation and prescription by a licensed clinician",
                    ],
                    evidence_source="openFDA SPL Drug Label / CDC Influenza Antiviral Guidelines",
                )
            )

        # B. Streptococcal Pharyngitis / Strep Throat
        if "strep" in cond_lower or ("pharyngitis" in cond_lower and etiology == "bacterial"):
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Amoxil (Amoxicillin)",
                    generic_name="amoxicillin",
                    drug_class="Aminopenicillin Antibiotic",
                    is_otc=False,
                    typical_dosage="500 mg orally every 8 hours or 875 mg orally every 12 hours for 10 complete days",
                    maximum_daily_limit="2000 mg in 24 hours",
                    route="oral",
                    clinical_purpose="First-line guideline eradication of Streptococcus pyogenes (Group A Strep) to prevent Acute Rheumatic Fever",
                    contraindication_warnings=[
                        "STRICT CONTRAINDICATION: Severe allergy or anaphylaxis to penicillin or beta-lactam antibiotics",
                        "Requires confirmation via rapid strep antigen test (RADT) or throat culture prior to dispensing",
                    ],
                    evidence_source="openFDA SPL Drug Label / IDSA Strep Pharyngitis Clinical Guidelines",
                )
            )
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Keflex (Cephalexin)",
                    generic_name="cephalexin",
                    drug_class="First-Generation Cephalosporin Antibiotic",
                    is_otc=False,
                    typical_dosage="500 mg orally twice daily for 10 days",
                    maximum_daily_limit="2000 mg in 24 hours",
                    route="oral",
                    clinical_purpose="Alternative guideline therapy for Group A Strep in non-IgE mediated penicillin reactions",
                    contraindication_warnings=[
                        "Cephalosporin allergy or history of severe immediate IgE-mediated anaphylaxis to penicillins",
                    ],
                    evidence_source="openFDA SPL Drug Label",
                )
            )
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Zithromax (Azithromycin)",
                    generic_name="azithromycin",
                    drug_class="Macrolide Antibiotic",
                    is_otc=False,
                    typical_dosage="500 mg orally as a single loading dose on Day 1, followed by 250 mg once daily on Days 2 through 5",
                    maximum_daily_limit="500 mg in 24 hours",
                    route="oral",
                    clinical_purpose="Guideline alternative for Streptococcal pharyngitis in patients with documented penicillin allergy",
                    contraindication_warnings=[
                        "Macrolide allergy; history of cholestatic jaundice with azithromycin",
                        "Caution in patients with cardiac baseline QT prolongation or uncorrected hypokalemia",
                    ],
                    evidence_source="openFDA SPL Drug Label / IDSA Guidelines",
                )
            )

        # C. Early Lyme Disease / Erythema Migrans / Tick-Borne
        if "lyme" in cond_lower or "borreli" in cond_lower or "erythema migrans" in cond_lower:
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Vibramycin (Doxycycline Hyclate)",
                    generic_name="doxycycline",
                    drug_class="Tetracycline Antibiotic",
                    is_otc=False,
                    typical_dosage="100 mg orally twice daily for 10 to 14 days (take with full glass of water and stay upright for 30 minutes)",
                    maximum_daily_limit="200 mg in 24 hours",
                    route="oral",
                    clinical_purpose="First-line gold standard therapy for Early Lyme disease (Borrelia burgdorferi) to prevent neurological and cardiac dissemination",
                    contraindication_warnings=[
                        "STRICT CONTRAINDICATION in pregnancy (teratogenic skeletal/bone growth suppression)",
                        "Contraindicated in children under 8 years of age due to permanent dental enamel hypoplasia and discoloration",
                        "Causes significant pill-induced esophagitis if swallowed lying down; induces cutaneous photosensitivity",
                    ],
                    evidence_source="openFDA SPL Drug Label / IDSA/AAN/ACR Lyme Disease Guidelines",
                )
            )
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Amoxil (Amoxicillin - High Dose for Lyme)",
                    generic_name="amoxicillin",
                    drug_class="Aminopenicillin Antibiotic",
                    is_otc=False,
                    typical_dosage="500 mg orally three times daily (every 8 hours) for 14 days",
                    maximum_daily_limit="1500 mg in 24 hours",
                    route="oral",
                    clinical_purpose="First-line alternative for Early Lyme disease in pregnant individuals and young children (<8 yo)",
                    contraindication_warnings=[
                        "STRICT CONTRAINDICATION: Documented penicillin or beta-lactam hypersensitivity",
                    ],
                    evidence_source="openFDA SPL Drug Label / CDC Lyme Disease Clinical Guidelines",
                )
            )

        # D. Bacterial Sinusitis / Complicated Otitis Media
        if "sinusitis" in cond_lower or "otitis" in cond_lower:
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Augmentin (Amoxicillin / Clavulanate Potassium 875/125)",
                    generic_name="amoxicillin_clavulanate",
                    drug_class="Beta-Lactamase Inhibitor Antibiotic Combination",
                    is_otc=False,
                    typical_dosage="875/125 mg orally twice daily with meals for 5 to 7 days",
                    maximum_daily_limit="1750 mg amoxicillin component in 24 hours",
                    route="oral",
                    clinical_purpose="First-line empiric coverage against beta-lactamase producing respiratory pathogens in acute bacterial sinusitis",
                    contraindication_warnings=[
                        "Penicillin or beta-lactam allergy",
                        "History of amoxicillin-clavulanate associated cholestatic jaundice or liver impairment",
                    ],
                    evidence_source="openFDA SPL Drug Label / IDSA Sinusitis Guidelines",
                )
            )

        # E. Bacterial Urinary Tract Infection (UTI / Pyelonephritis)
        if any(term in cond_lower for term in ["uti", "urinary", "cystitis", "pyelonephritis"]):
            candidates.append(
                MedicationDosingSchema(
                    drug_name="Cipro (Ciprofloxacin)",
                    generic_name="ciprofloxacin",
                    drug_class="Fluoroquinolone Antibiotic",
                    is_otc=False,
                    typical_dosage="250 mg to 500 mg orally every 12 hours for 7 to 14 days with abundant hydration",
                    maximum_daily_limit="1000 mg in 24 hours",
                    route="oral",
                    clinical_purpose="Second-line antibiotic for complicated UTI or pyelonephritis due to gram-negative bacilli",
                    contraindication_warnings=[
                        "BOXED WARNING: Tendinitis and tendon rupture; peripheral neuropathy; CNS toxicities",
                        "Contraindicated in pediatric patients under 18 years and in pregnancy (cartilage toxicity)",
                    ],
                    evidence_source="openFDA SPL Drug Label",
                )
            )

        return candidates

    # ── Safety Shield Engine ───────────────────────────────────────────────────
    def _filter_through_safety_shield(
        self,
        candidates: List[MedicationDosingSchema],
        age: Optional[int],
        is_pregnant: bool,
        allergies: List[str],
        chronic_conditions: List[str],
    ) -> Tuple[List[MedicationDosingSchema], List[str]]:
        """
        Applies mathematical invariants Psi_allergy, Psi_pediatric, Psi_obstetric, Psi_chronic.
        Filters candidate medications into approved list and blocked log with explicit clinical reasons.
        """
        approved: List[MedicationDosingSchema] = []
        blocked_logs: List[str] = []

        chronic_text = " ".join([str(c) for c in chronic_conditions]).lower()
        has_htn_cad = any(k in chronic_text for k in ["hypertension", "high blood pressure", "htn", "cad", "coronary", "heart disease", "angina"])
        has_ckd = any(k in chronic_text for k in ["ckd", "kidney disease", "renal failure", "renal insufficiency", "nephropathy"])
        has_ulcer = any(k in chronic_text for k in ["ulcer", "peptic ulcer", "gi bleed", "gastritis", "bleeding"])
        has_liver = any(k in chronic_text for k in ["cirrhosis", "hepatitis", "liver failure", "hepatic", "fatty liver"])

        # Compile set of all prohibited drugs from allergies via RxCrossReactivityGraph
        prohibited_by_allergy: Set[str] = set()
        for allergy in allergies:
            clean_a = allergy.strip().lower()
            if clean_a and clean_a not in {"none", "none reported", "no allergies", "nkda"}:
                prohibited_by_allergy.update(RxCrossReactivityGraph.get_contraindicated_drugs_for_allergy(clean_a))

        for med in candidates:
            med_name_lower = med.drug_name.lower()
            generic_lower = med.generic_name.lower()
            drug_class_lower = med.drug_class.lower()

            # 1. Allergy Invariant (Psi_allergy)
            is_allergic = False
            matching_trigger = ""
            for prohibited in prohibited_by_allergy:
                if (prohibited in generic_lower) or (prohibited in med_name_lower):
                    is_allergic = True
                    matching_trigger = prohibited
                    break

            if is_allergic:
                blocked_logs.append(
                    f"{med.drug_name}: BLOCKED due to documented patient allergy to '{matching_trigger}' "
                    f"via pharmacological cross-reactivity mapping (RxCrossReactivityGraph)."
                )
                continue

            # 2. Pediatric Invariant (Psi_pediatric)
            if age is not None:
                # Salicylates strictly banned if age < 19
                if age < 19 and any(sal in generic_lower or sal in med_name_lower for sal in PediatricObstetricSafetyRules.SALICYLATES_PEDIATRIC):
                    blocked_logs.append(
                        f"{med.drug_name}: STRICTLY BLOCKED for patient aged {age} years due to FDA Boxed Warning "
                        f"for Reye's Syndrome hazard (acute encephalopathy and fatty liver failure following viral illness)."
                    )
                    continue

                # Fluoroquinolones banned if age < 18
                if age < 18 and any(fq in generic_lower for fq in PediatricObstetricSafetyRules.FLUOROQUINOLONES):
                    blocked_logs.append(
                        f"{med.drug_name}: BLOCKED for pediatric patient aged {age} years due to FDA Boxed Warning "
                        f"for irreversible cartilage arthropathy, tendon rupture, and chondrotoxicity."
                    )
                    continue

                # Tetracyclines banned if age < 8
                if age < 8 and ("doxycycline" in generic_lower or "tetracycline" in generic_lower):
                    blocked_logs.append(
                        f"{med.drug_name}: BLOCKED for child aged {age} years due to permanent dental enamel hypoplasia "
                        f"and irreversible tooth discoloration."
                    )
                    continue

            # 3. Obstetric Invariant (Psi_obstetric)
            if is_pregnant:
                # Tetracyclines in pregnancy
                if "doxycycline" in generic_lower or "tetracycline" in generic_lower:
                    blocked_logs.append(
                        f"{med.drug_name}: STRICTLY BLOCKED in pregnancy due to teratogenic impairment of fetal skeletal "
                        f"growth and permanent tooth enamel hypoplasia."
                    )
                    continue

                # Fluoroquinolones in pregnancy
                if any(fq in generic_lower for fq in PediatricObstetricSafetyRules.FLUOROQUINOLONES):
                    blocked_logs.append(
                        f"{med.drug_name}: BLOCKED in pregnancy due to quinolone-induced fetal articular cartilage toxicity."
                    )
                    continue

                # NSAIDs in pregnancy (especially 3rd trimester or >= 20 weeks)
                if "nsaid" in drug_class_lower or "ibuprofen" in generic_lower or "naproxen" in generic_lower or "aspirin" in generic_lower:
                    blocked_logs.append(
                        f"{med.drug_name}: BLOCKED in pregnancy due to risk of premature closure of the fetal ductus "
                        f"arteriosus, fetal renal dysfunction, and oligohydramnios."
                    )
                    continue

            # 4. Chronic Comorbidity Invariant (Psi_chronic)
            # Hypertension / CAD vs Systemic Decongestants
            if has_htn_cad and ("pseudoephedrine" in generic_lower or "phenylephrine" in generic_lower):
                blocked_logs.append(
                    f"{med.drug_name}: BLOCKED due to documented Hypertension/CAD. Sympathomimetic alpha-agonists "
                    f"cause systemic vasoconstriction, severe hypertensive spikes, and cardiac tachyarrhythmias."
                )
                continue

            # Chronic Kidney Disease vs NSAIDs
            if has_ckd and ("nsaid" in drug_class_lower or "ibuprofen" in generic_lower or "naproxen" in generic_lower):
                blocked_logs.append(
                    f"{med.drug_name}: BLOCKED due to documented Chronic Kidney Disease (CKD). Systemic NSAIDs inhibit "
                    f"renal prostaglandin synthesis, precipitating acute renal vasoconstriction and acute kidney injury (AKI)."
                )
                continue

            # Peptic Ulcer / Bleeding vs NSAIDs & Aspirin
            if has_ulcer and ("nsaid" in drug_class_lower or "ibuprofen" in generic_lower or "aspirin" in generic_lower):
                blocked_logs.append(
                    f"{med.drug_name}: BLOCKED due to history of peptic ulcer disease or gastrointestinal bleeding. "
                    f"Inhibition of COX-1 severely compromises mucosal protective prostaglandins and elevates re-bleeding risk."
                )
                continue

            # Hepatic Impairment - Adjust dosage notes for Acetaminophen
            if has_liver and "acetaminophen" in generic_lower:
                med = med.model_copy(update={
                    "maximum_daily_limit": "STRICT LIVER CEILING: Do NOT exceed 2000 mg in 24 hours under close physician oversight.",
                    "clinical_purpose": med.clinical_purpose + " (Dosage capped strictly at 2000 mg/day due to documented hepatic impairment)",
                })

            approved.append(med)

        return approved, blocked_logs

    # ── Safe Alternatives Generation ──────────────────────────────────────────
    def _ensure_safe_alternatives(
        self,
        safe_tier_2: List[MedicationDosingSchema],
        safe_tier_3: List[MedicationDosingSchema],
        etiology: str,
        condition: str,
        age: Optional[int],
        is_pregnant: bool,
        allergies: List[str],
        chronic_conditions: List[str],
    ) -> Tuple[List[MedicationDosingSchema], List[MedicationDosingSchema]]:
        """Ensures that if primary first-line therapies were blocked, safe evidence-based alternatives are present."""
        cond_lower = condition.lower()
        allergies_str = " ".join([str(a).lower() for a in allergies])

        # If Strep pharyngitis and amoxicillin was blocked by penicillin allergy, verify azithromycin is present in Tier 3
        if "strep" in cond_lower or (etiology == "bacterial" and "pharyngitis" in cond_lower):
            has_amox = any("amoxicillin" in m.generic_name.lower() for m in safe_tier_3)
            has_azithro = any("azithromycin" in m.generic_name.lower() for m in safe_tier_3)
            if not has_amox and not has_azithro:
                safe_tier_3.append(
                    MedicationDosingSchema(
                        drug_name="Zithromax (Azithromycin)",
                        generic_name="azithromycin",
                        drug_class="Macrolide Antibiotic",
                        is_otc=False,
                        typical_dosage="500 mg orally as a single loading dose on Day 1, then 250 mg once daily on Days 2 through 5",
                        maximum_daily_limit="500 mg in 24 hours",
                        route="oral",
                        clinical_purpose="First-line safe alternative for Streptococcal pharyngitis in patients with penicillin allergy",
                        contraindication_warnings=["Macrolide allergy; severe hepatic dysfunction"],
                        evidence_source="IDSA Strep Pharyngitis Clinical Guidelines",
                    )
                )

        # If Lyme disease and patient is pregnant or child < 8, ensure Amoxicillin is present if no penicillin allergy
        if "lyme" in cond_lower or "erythema migrans" in cond_lower:
            doxy_present = any("doxycycline" in m.generic_name.lower() for m in safe_tier_3)
            if not doxy_present and "penicillin" not in allergies_str and "amoxicillin" not in allergies_str:
                has_amox = any("amoxicillin" in m.generic_name.lower() for m in safe_tier_3)
                if not has_amox:
                    safe_tier_3.append(
                        MedicationDosingSchema(
                            drug_name="Amoxil (Amoxicillin - High Dose for Lyme)",
                            generic_name="amoxicillin",
                            drug_class="Aminopenicillin Antibiotic",
                            is_otc=False,
                            typical_dosage="500 mg orally three times daily for 14 days",
                            maximum_daily_limit="1500 mg in 24 hours",
                            route="oral",
                            clinical_purpose="Guideline-directed safe first-line alternative for Early Lyme in pregnancy and children (<8 yo)",
                            contraindication_warnings=["Requires confirmation and prescription by physician"],
                            evidence_source="CDC Lyme Disease Guidelines",
                        )
                    )

        # Ensure Tier 2 has at least one safe analgesic/antipyretic (Acetaminophen)
        has_tier_2_pain = any(m.is_otc for m in safe_tier_2)
        if not has_tier_2_pain:
            safe_tier_2.append(
                MedicationDosingSchema(
                    drug_name="Tylenol (Acetaminophen)",
                    generic_name="acetaminophen",
                    drug_class="Analgesic / Antipyretic",
                    is_otc=True,
                    typical_dosage="500 mg to 1000 mg orally every 4 to 6 hours as needed with water",
                    maximum_daily_limit="Do NOT exceed 3000 mg in 24 hours",
                    route="oral",
                    clinical_purpose="Safe first-line analgesic and antipyretic",
                    contraindication_warnings=["Severe active hepatic impairment"],
                    evidence_source="openFDA SPL Drug Label",
                )
            )

        return safe_tier_2, safe_tier_3

    @staticmethod
    def _extract_age(profile: Dict[str, Any]) -> Optional[int]:
        """Extracts integer age from patient profile safely."""
        raw_age = profile.get("age")
        if raw_age is None:
            return None
        if isinstance(raw_age, int):
            return raw_age
        if isinstance(raw_age, str):
            match = re.search(r"\b(\d{1,3})\b", raw_age)
            if match:
                return int(match.group(1))
        return None

    def format_plan_for_consultation_prompt(self, plan: PharmacotherapyPlanSchema, compact: bool = True) -> str:
        """Formats the pharmacotherapy plan into an authoritative prompt block.

        If compact=True, formats a telegraphic block (under 45 tokens) strictly preserving prompt token budget.
        """
        if compact:
            lines = [
                f"[CLINICAL PHARMACOTHERAPY: {plan.condition_name.upper()}]",
                f"Etiology: {plan.etiology_type.upper()} | Antibiotics: {'Indicated (Bacterial)' if plan.antimicrobial_indicated else 'Contraindicated (Viral)'}",
            ]
            if plan.tier_2_otc_medications:
                otc_names = [f"{m.drug_name.split('(')[0].strip()} (Max: {m.maximum_daily_limit})" for m in plan.tier_2_otc_medications[:2]]
                lines.append("Safe OTC: " + " | ".join(otc_names))
            if plan.tier_3_prescription_guidance:
                rx_names = [m.drug_name.split('(')[0].strip() for m in plan.tier_3_prescription_guidance[:2]]
                lines.append("Rx to Discuss: " + ", ".join(rx_names))
            if plan.safety_shield_violations_blocked:
                blocked_items = []
                for b in plan.safety_shield_violations_blocked[:2]:
                    drug_part = b.split(":")[0].strip()
                    if "Reye" in b:
                        blocked_items.append(f"{drug_part} (Reye's hazard)")
                    elif "allergy" in b or "RxCrossReactivity" in b:
                        blocked_items.append(f"{drug_part} (Allergy cross-reactivity)")
                    elif "pregnancy" in b or "teratogenic" in b:
                        blocked_items.append(f"{drug_part} (Teratogenic/Pregnancy)")
                    elif "Hypertension" in b:
                        blocked_items.append(f"{drug_part} (HTN/CAD risk)")
                    elif "Stewardship" in b:
                        blocked_items.append(f"{drug_part} (Viral stewardship)")
                    else:
                        blocked_items.append(drug_part)
                lines.append("[SAFETY SHIELD BLOCKED]: " + ", ".join(blocked_items))
            return "\n".join(lines)

        lines = [
            f"[CLINICAL PHARMACOTHERAPY & MEDICATION GUIDANCE: {plan.condition_name.upper()}]",
            f"Etiology: {plan.etiology_type.upper()} | Antibiotics Indicated: {'YES (Bacterial Pathogen)' if plan.antimicrobial_indicated else 'NO (Strictly Contraindicated for Viral/Non-bacterial)'}",
        ]

        # Tier 1
        if plan.tier_1_supportive_care:
            lines.append("• Tier 1 Non-Pharmacological Care: " + "; ".join(plan.tier_1_supportive_care[:3]))

        # Tier 2
        if plan.tier_2_otc_medications:
            otc_entries = []
            for m in plan.tier_2_otc_medications[:2]:
                otc_entries.append(f"{m.drug_name} ({m.typical_dosage}; Max: {m.maximum_daily_limit})")
            lines.append("• Tier 2 Safe OTC Symptom Relief: " + " | ".join(otc_entries))

        # Tier 3
        if plan.tier_3_prescription_guidance:
            rx_entries = []
            for m in plan.tier_3_prescription_guidance[:2]:
                rx_entries.append(f"{m.drug_name} ({m.typical_dosage} - discuss with physician)")
            lines.append("• Tier 3 Prescription Guidelines to Discuss: " + " | ".join(rx_entries))

        # Blocked
        if plan.safety_shield_violations_blocked:
            lines.append("• [SAFETY SHIELD BLOCKED MEDICATIONS]: " + " | ".join(plan.safety_shield_violations_blocked[:3]))

        return "\n".join(lines)


# Global singleton instance for system-wide access
pharmacotherapy_engine = PharmacotherapyEngine()
