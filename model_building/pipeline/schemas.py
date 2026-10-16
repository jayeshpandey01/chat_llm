"""
Pydantic V2 Data Contracts for Clinical Symptom-to-Disease Prediction Pipeline.
==============================================================================
Defines strict input requests, evidence groundings, condition explanations,
and comprehensive diagnostic reports conforming to FDA CDSS guidelines.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class SymptomInputRequest(BaseModel):
    """Input payload accepting free-text, structured checklist tokens, and demographics."""
    
    query_text: Optional[str] = Field(
        default=None,
        description="Free-text narrative symptom description (e.g., 'Severe cough and fever for 2 days, denies headache')"
    )
    selected_symptoms: Optional[List[str]] = Field(
        default=None,
        description="Structured list of symptom codes from onboarding checklist (e.g., ['cough', 'wheezing'])"
    )
    age: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=125.0,
        description="Patient age in years (used for demographic and biological rule filtering)"
    )
    biological_sex: Optional[str] = Field(
        default=None,
        description="Patient biological sex ('MALE', 'FEMALE', or 'INTERSEX')"
    )
    is_pregnant: Optional[bool] = Field(
        default=False,
        description="Whether patient is currently pregnant (triggers obstetric differential rules)"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of differential conditions to return in the report"
    )
    conformal_coverage: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description="Mathematical confidence coverage level for split conformal prediction set"
    )
    model_tier: Optional[str] = Field(
        default="auto",
        description="Target model architecture: 'auto', 'primary_41', 'general_50', or 'all_580'"
    )


class MatchedSymptomEvidence(BaseModel):
    """Quantitative evidence associating an individual symptom with a candidate condition."""
    
    symptom: str = Field(..., description="Canonical symptom token")
    diagnostic_odds_ratio: float = Field(..., description="Empirical Diagnostic Odds Ratio (DOR) from clinical records")
    p_s_given_disease: float = Field(..., description="Probability of symptom presentation given the disease")
    is_pathognomonic: bool = Field(default=False, description="Whether symptom is a defining pathognomonic hallmark")


class ConditionExplanation(BaseModel):
    """Explainable diagnostic candidate with symptom attribution and NIH MedQuAD citations."""
    
    condition: str = Field(..., description="Standardized clinical condition name")
    probability: float = Field(..., description="Calibrated posterior probability (0.0 to 1.0)")
    confidence_percentage: float = Field(..., description="Confidence percentage formatted for patient UI (0.0 to 100.0)")
    rank: int = Field(..., ge=1, description="Rank within differential diagnosis list")
    source_model: str = Field(..., description="Underlying model producing this prediction")
    matched_symptoms: List[str] = Field(default_factory=list, description="Patient symptoms that directly support this diagnosis")
    missing_classic_symptoms: List[str] = Field(default_factory=list, description="Common hallmark symptoms not reported by patient")
    evidence_symptoms: List[MatchedSymptomEvidence] = Field(default_factory=list, description="Detailed odds ratio evidence")
    explanation_text: str = Field(..., description="Patient-friendly plain language clinical explanation")
    nih_guideline_topic: Optional[str] = Field(default=None, description="Topic title from NIH / MedlinePlus guidelines")
    nih_guideline_summary: Optional[str] = Field(default=None, description="Clinical summary excerpt from NIH / MedlinePlus")


class EmergencyAlert(BaseModel):
    """Critical emergency acuity alert triggered by life-threatening symptom patterns."""
    
    is_emergency: bool = Field(..., description="Whether immediate emergency escalation is required")
    acuity_level: int = Field(..., ge=1, le=5, description="Emergency Severity Index level (1=Resuscitation, 2=Emergent)")
    condition: str = Field(..., description="Suspected emergency condition (e.g. Myocardial Infarction, Stroke)")
    action: str = Field(..., description="Mandatory triage action (e.g. 'CALL_911_IMMEDIATELY')")
    message: str = Field(..., description="Urgent patient-facing emergency instruction")


class ActiveQuestionInquiry(BaseModel):
    """High-yield Bayesian follow-up questions to resolve non-specific solitary symptoms."""
    
    is_active_inquiry_needed: bool = Field(..., description="Whether symptom entropy requires follow-up questioning")
    primary_symptom: str = Field(..., description="Non-specific presentation symptom (e.g. 'fatigue')")
    candidate_conditions: List[str] = Field(default_factory=list, description="Potential conditions in competition")
    high_yield_questions: List[str] = Field(default_factory=list, description="Top discriminative questions to ask next")
    rationale: str = Field(..., description="Clinical reasoning explaining why these questions are needed")


class DiagnosticReportSchema(BaseModel):
    """Comprehensive diagnostic report produced by the clinical prediction pipeline."""
    
    query_text: str = Field(..., description="Original input query or description")
    extracted_present_symptoms: List[str] = Field(default_factory=list, description="Extracted affirmed clinical symptoms")
    extracted_denied_symptoms: List[str] = Field(default_factory=list, description="Extracted denied / ruled-out symptoms (NegEx)")
    top_condition: str = Field(..., description="Primary candidate condition name")
    top_confidence: float = Field(..., description="Confidence percentage of the primary candidate")
    differential_diagnosis: List[ConditionExplanation] = Field(default_factory=list, description="Ranked differential list with explanations")
    conformal_differential_set: List[str] = Field(default_factory=list, description="Adaptive prediction set guaranteeing 95% coverage")
    conformal_coverage_guaranteed: bool = Field(default=True, description="Whether mathematical coverage bound is active")
    emergency_alert: Optional[EmergencyAlert] = Field(default=None, description="Emergency life-threat alert if triggered")
    active_inquiry: Optional[ActiveQuestionInquiry] = Field(default=None, description="Active clarifying questions if needed")
    demographic_exclusions: List[str] = Field(default_factory=list, description="Diseases excluded by biological age/sex rules")
    is_out_of_distribution: bool = Field(default=False, description="Whether query was non-medical or conversational")
    abstention_triggered: bool = Field(default=False, description="Whether pipeline abstained from single point diagnosis")
    triage_level: str = Field(default="ESI-4", description="Overall triage urgency tier (ESI-1 to ESI-5)")
    triage_recommendation: str = Field(..., description="Actionable clinical recommendation for next steps")
    cached: bool = Field(default=False, description="Whether response was served from high-speed canonical cache")
    execution_time_ms: float = Field(default=0.0, description="Total pipeline latency in milliseconds")
    disclaimer: str = Field(
        default=(
            "HealthTrack AI is a clinical preliminary decision support tool and not a substitute for professional "
            "medical diagnosis, prognosis, or treatment. If you are experiencing severe chest pain, sudden numbness, "
            "or difficulty breathing, please seek immediate emergency care (call 911 / 112)."
        ),
        description="Statutory FDA CDSS clinical disclaimer"
    )

