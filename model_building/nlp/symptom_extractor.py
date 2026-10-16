"""
Clinical Natural Language Extraction & Normalization Engine
===========================================================
Implements:
1. Lexical tokenization & medical phrase n-gram extraction (1-gram to 4-gram).
2. Rule-based clinical negation detection (NegEx paradigm: "no", "denies", "without", "absent").
3. Levenshtein edit distance & character n-gram fuzzy matching for typo recovery.
4. Inflection / lemmatization normalization: maps "coughing" → "cough", "vomiting" → "vomit", etc.
   This fixes the critical hallucination gap where symptom variants were missed.
5. Mapping of free-text patient utterances into canonical feature space tokens.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Clinical negation triggers ─────────────────────────────────────────────────
NEGATION_PRE_TRIGGERS = [
    "no",
    "not",
    "none",
    "denies",
    "denied",
    "denying",
    "without",
    "negative for",
    "free of",
    "rules out",
    "ruled out",
    "absence of",
    "never had",
    "no sign of",
    "no signs of",
    "no complaints of",
]

NEGATION_POST_TRIGGERS = [
    "was ruled out",
    "is ruled out",
    "unlikely",
    "negative",
]

PSEUDO_NEGATIONS = [
    "no doubt",
    "not only",
    "no further",
    "without delay",
]

CONVERSATIONAL_STOP_WORDS = {
    "patient",
    "doctor",
    "severe",
    "mild",
    "moderate",
    "experiencing",
    "states",
    "feeling",
    "having",
    "present",
    "presents",
    "reported",
    "reports",
    "today",
    "yesterday",
    "chronic",
    "acute",
    "history",
}

# ── Inflection / Lemmatization map ─────────────────────────────────────────────
# Maps common surface forms (progressive, past, plural, informal synonyms) to their
# canonical vocabulary token or a close canonical equivalent.
# This directly fixes the "coughing ≠ cough" extraction gap.
INFLECTION_MAP: dict[str, str] = {
    # Respiratory
    "coughing": "cough",
    "coughed": "cough",
    "sneezing": "continuous_sneezing",
    "sneeze": "continuous_sneezing",
    "breathlessness": "breathlessness",
    "shortness of breath": "breathlessness",
    "short of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "breathing difficulty": "breathlessness",
    "wheezing": "breathlessness",
    # Gastrointestinal
    "vomiting": "vomiting",
    "vomited": "vomiting",
    "puking": "vomiting",
    "nauseous": "nausea",
    "nauseated": "nausea",
    "diarrhea": "diarrhoea",
    "diarrheal": "diarrhoea",
    "loose stools": "diarrhoea",
    "loose motions": "diarrhoea",
    # Fever / temperature
    "feverish": "mild_fever",
    "running a fever": "mild_fever",
    "high temperature": "high_fever",
    "very high fever": "high_fever",
    "elevated temperature": "mild_fever",
    # Pain
    "aching": "pain_behind_the_eyes",  # will be overridden by n-gram if more specific
    "aches": "body_ache",
    "body aches": "body_ache",
    "muscle ache": "muscle_pain",
    "muscle aches": "muscle_pain",
    "joint ache": "joint_pain",
    "joint aches": "joint_pain",
    "back ache": "back_pain",
    "backache": "back_pain",
    "headaches": "headache",
    "migraines": "headache",
    # Skin
    "itching": "itching",
    "itchy": "itching",
    "itchy skin": "skin_rash",
    "rashing": "skin_rash",
    "rashes": "skin_rash",
    "spotting": "spotting_urination",
    # Fatigue / weakness
    "fatigued": "fatigue",
    "exhausted": "fatigue",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "lethargic": "lethargy",
    "lethargy": "lethargy",
    "weakness": "weakness_in_limbs",
    "weak": "weakness_in_limbs",
    # Eyes
    "yellowing of eyes": "yellowing_of_eyes",
    "yellow eyes": "yellowing_of_eyes",
    "jaundiced eyes": "yellowing_of_eyes",
    "watery eyes": "watering_from_eyes",
    "eye discharge": "watering_from_eyes",
    # Urinary
    "burning urination": "burning_micturition",
    "painful urination": "burning_micturition",
    "frequent urination": "polyuria",
    "excessive urination": "polyuria",
    # Appetite
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "poor appetite": "loss_of_appetite",
    # Throat
    "sore throat": "throat_irritation",
    "throat pain": "throat_irritation",
    "strep throat": "throat_irritation",
    # Abdominal
    "stomach pain": "stomach_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "abdominal_pain",
    "stomach cramps": "stomach_pain",
    # Chest
    "chest tightness": "chest_pain",
    "chest pressure": "chest_pain",
    "chest tightening": "chest_pain",
    # Other common
    "sweating": "sweating",
    "excessive sweating": "sweating",
    "night sweats": "sweating",
    "chills": "chills",
    "shivering": "shivering",
    "rigors": "shivering",
    "swollen glands": "swollen_blood_vessels",
    "lymph node swelling": "swollen_blood_vessels",
    "blurred vision": "blurred_and_distorted_vision",
    "blurry vision": "blurred_and_distorted_vision",
    "distorted vision": "blurred_and_distorted_vision",
    "double vision": "blurred_and_distorted_vision",
    "skin peeling": "peeling_skin",
    "skin flaking": "peeling_skin",
    "mood swings": "mood_swings",
    "irritable": "irritability",
    "irritability": "irritability",
    "depression": "depression",
    "depressed": "depression",
    "anxious": "anxiety",
    "palpitations": "heart_palpitations",
    "racing heart": "heart_palpitations",
    "rapid heartbeat": "heart_palpitations",
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings with O(min(N,M)) memory."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Computes normalized Levenshtein similarity ratio between 0.0 and 1.0."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


@dataclass
class ExtractedSymptom:
    canonical_token: str
    original_text: str
    is_negated: bool = False
    confidence: float = 1.0
    match_type: str = "exact"  # "exact", "inflection", "ngram_phrase", "fuzzy"


class ClinicalSymptomExtractor:
    """
    Robust clinical symptom extractor for natural language patient inputs.
    Uses clinical vocabulary dictionaries, n-gram sliding windows, NegEx-style
    negation detection, inflection/lemmatization normalization, and fuzzy typo matching.
    """

    def __init__(self, vocab_path: Path | None = None, fuzzy_threshold: float = 0.82):
        self.fuzzy_threshold = fuzzy_threshold
        self.canonical_symptoms: set[str] = set()
        self.phrase_to_canonical: dict[str, str] = {}

        if vocab_path and vocab_path.exists():
            self.load_vocabulary(vocab_path)

    def load_vocabulary(self, vocab_path: Path):
        """Loads canonical symptom vocabulary from JSON and indexes fast lookups."""
        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)
            raw_symptoms = data.get("symptoms", [])

        self.canonical_symptoms = set(raw_symptoms)
        for s in self.canonical_symptoms:
            # Map canonical snake_case to natural spaced phrase
            spaced = s.replace("_", " ")
            self.phrase_to_canonical[spaced] = s
            self.phrase_to_canonical[s] = s

        # Register all inflection/lemmatization mappings that resolve to a known canonical token
        for surface_form, canonical_target in INFLECTION_MAP.items():
            # Only register if the canonical_target exists in vocabulary
            if canonical_target in self.canonical_symptoms:
                self.phrase_to_canonical[surface_form] = canonical_target
                # Also register underscored version
                underscored = surface_form.replace(" ", "_")
                self.phrase_to_canonical[underscored] = canonical_target
            else:
                # Fallback: register anyway; extractor will flag with inflection match type
                self.phrase_to_canonical[surface_form] = canonical_target

        logger.info(
            f"Loaded {len(self.canonical_symptoms)} canonical symptoms into NLP index "
            f"(+ {len(INFLECTION_MAP)} inflection mappings)."
        )

    def add_custom_synonym(self, phrase: str, canonical_token: str):
        """Registers custom colloquial or clinical synonym mapping."""
        norm_phrase = phrase.strip().lower()
        self.phrase_to_canonical[norm_phrase] = canonical_token

    def _split_into_sentences(self, text: str) -> list[str]:
        """Splits free text into clinical sentence or clause chunks."""
        clean_text = text.replace("\r", " ")
        clauses = re.split(r"[.\n;!?,]+", clean_text)
        return [c.strip().lower() for c in clauses if c.strip()]

    def _detect_negation(self, clause: str, target_phrase: str) -> bool:
        """
        Determines whether target_phrase in clause is in the scope of a negation trigger.
        Follows the clinical NegEx standard with a 5-token left scope window.
        """
        # Remove pseudo negations first
        for p_neg in PSEUDO_NEGATIONS:
            if p_neg in clause:
                clause = clause.replace(p_neg, "")

        phrase_pos = clause.find(target_phrase)
        if phrase_pos == -1:
            return False

        # Pre-negation scope (up to 5 tokens before the target phrase)
        pre_text = clause[:phrase_pos].strip()
        pre_tokens = pre_text.split()
        pre_window = " ".join(pre_tokens[-5:]).lower() if pre_tokens else ""

        for trigger in NEGATION_PRE_TRIGGERS:
            pattern = rf"\b{re.escape(trigger)}\b"
            if re.search(pattern, pre_window):
                return True

        # Post-negation scope (up to 3 tokens after the target phrase)
        post_text = clause[phrase_pos + len(target_phrase):].strip()
        post_tokens = post_text.split()
        post_window = " ".join(post_tokens[:3]).lower() if post_tokens else ""

        for trigger in NEGATION_POST_TRIGGERS:
            pattern = rf"\b{re.escape(trigger)}\b"
            if re.search(pattern, post_window):
                return True

        return False

    def extract_symptoms(self, free_text: str) -> list[ExtractedSymptom]:
        """
        Extracts all canonical clinical symptoms present in free-text input,
        correcting inflections, typos and flagging negated signs.

        Matching priority:
        1. Longest n-gram exact/inflection match (4→1 grams)
        2. Fuzzy match for remaining unmatched tokens (Levenshtein similarity >= threshold)
        """
        if not free_text or not free_text.strip():
            return []

        clauses = self._split_into_sentences(free_text)
        extracted_map: dict[str, ExtractedSymptom] = {}

        for clause in clauses:
            tokens = clause.split()
            n_tokens = len(tokens)
            matched_spans: list[tuple[int, int]] = []

            # 1. Multi-token n-gram search (4-grams down to 1-grams)
            for n in range(min(4, n_tokens), 0, -1):
                for i in range(n_tokens - n + 1):
                    if any(s <= i and i + n <= e for s, e in matched_spans):
                        continue

                    candidate_phrase = " ".join(tokens[i: i + n])
                    canonical = self.phrase_to_canonical.get(candidate_phrase)

                    if canonical:
                        match_type = "inflection" if candidate_phrase in INFLECTION_MAP else "exact"
                        is_neg = self._detect_negation(clause, candidate_phrase)
                        if canonical not in extracted_map or (
                            extracted_map[canonical].is_negated and not is_neg
                        ):
                            extracted_map[canonical] = ExtractedSymptom(
                                canonical_token=canonical,
                                original_text=candidate_phrase,
                                is_negated=is_neg,
                                confidence=1.0,
                                match_type=match_type,
                            )
                        matched_spans.append((i, i + n))

            # 2. Fuzzy matching for unmatched tokens/bigrams
            for i in range(n_tokens):
                if any(s <= i < e for s, e in matched_spans):
                    continue

                for n in [1, 2]:
                    if i + n > n_tokens:
                        continue
                    unmatched_phrase = " ".join(tokens[i: i + n])
                    if (
                        len(unmatched_phrase) < 4
                        or unmatched_phrase in CONVERSATIONAL_STOP_WORDS
                    ):
                        continue

                    best_match = None
                    best_score = 0.0

                    for phrase, canonical in self.phrase_to_canonical.items():
                        if abs(len(phrase) - len(unmatched_phrase)) > 3:
                            continue
                        score = similarity_ratio(unmatched_phrase, phrase)
                        if score > best_score:
                            best_score = score
                            best_match = canonical

                    if best_match and best_score >= self.fuzzy_threshold:
                        is_neg = self._detect_negation(clause, unmatched_phrase)
                        if best_match not in extracted_map:
                            extracted_map[best_match] = ExtractedSymptom(
                                canonical_token=best_match,
                                original_text=unmatched_phrase,
                                is_negated=is_neg,
                                confidence=round(best_score, 3),
                                match_type="fuzzy",
                            )
                        matched_spans.append((i, i + n))

        return list(extracted_map.values())

    def extract_positive_symptoms(self, free_text: str) -> list[str]:
        """Convenience method returning list of affirmed canonical tokens only."""
        results = self.extract_symptoms(free_text)
        return [r.canonical_token for r in results if not r.is_negated]
