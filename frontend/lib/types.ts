/**
 * HealthTrack AI Data Contracts (mirrors backend Pydantic models)
 */

export interface SymptomInputRequest {
  query_text?: string;
  selected_symptoms?: string[];
  age?: number | null;
  biological_sex?: string | null;
  is_pregnant?: boolean;
  top_k?: number;
  conformal_coverage?: number;
  model_tier?: "auto" | "primary_41" | "general_50" | "all_580";
}

export interface MatchedSymptomEvidence {
  symptom: string;
  diagnostic_odds_ratio: number;
  p_s_given_disease: number;
  is_pathognomonic: boolean;
}

export interface ConditionExplanation {
  condition: string;
  probability: number;
  confidence_percentage: number;
  rank: number;
  source_model: string;
  matched_symptoms: string[];
  missing_classic_symptoms: string[];
  evidence_symptoms: MatchedSymptomEvidence[];
  explanation_text: string;
  nih_guideline_topic?: string | null;
  nih_guideline_summary?: string | null;
}

export interface EmergencyAlert {
  is_emergency: boolean;
  acuity_level: number;
  condition: string;
  action: string;
  message: string;
}

export interface ActiveQuestionInquiry {
  is_active_inquiry_needed: boolean;
  primary_symptom: string;
  candidate_conditions: string[];
  high_yield_questions: string[];
  rationale: string;
}

export interface DiagnosticReportSchema {
  query_text: string;
  extracted_present_symptoms: string[];
  extracted_denied_symptoms: string[];
  top_condition: string;
  top_confidence: number;
  differential_diagnosis: ConditionExplanation[];
  conformal_differential_set: string[];
  conformal_coverage_guaranteed: boolean;
  emergency_alert?: EmergencyAlert | null;
  active_inquiry?: ActiveQuestionInquiry | null;
  demographic_exclusions: string[];
  is_out_of_distribution: boolean;
  abstention_triggered: boolean;
  triage_level: string;
  triage_recommendation: string;
  cached: boolean;
  execution_time_ms: number;
  disclaimer: string;
}

export interface MemberProfile {
  member_id: string;
  uid: string;
  name: string;
  relationship: string;
  age?: number | null;
  biological_sex?: string | null;
  is_pregnant: boolean;
  chronic_conditions: string[];
  current_medications: string[];
  drug_allergies: string[];
  baseline_symptoms: string[];
  longitudinal_summary: Array<{
    condition: string;
    icd10?: string;
    diagnosed_date?: string;
    status: string;
    notes?: string;
    session_id?: string;
    recorded_at?: number;
  }>;
  created_at?: number;
  updated_at?: number;
}

export interface SessionRecord {
  session_id: string;
  uid: string;
  member_id: string;
  member_name: string;
  member_relationship: string;
  title: string;
  status: string;
  created_at: number;
  updated_at: number;
  recent_context_window: Array<{
    turn: number;
    user: string;
    assistant: string;
    top_condition?: string;
    timestamp?: number;
  }>;
  active_differential?: DiagnosticReportSchema | null;
}

export interface SSEMetadataEvent {
  type: "diagnostic_report";
  member_id: string;
  member_name: string;
  member_relationship: string;
  top_condition: string;
  top_confidence: number;
  triage_level: string;
  triage_recommendation: string;
  conformal_set: string[];
  is_emergency: boolean;
  is_acute_on_chronic: boolean;
  acute_on_chronic_alert?: string | null;
  retrieved_evidence?: Array<{
    condition: string;
    source_title?: string;
    source_type: string;
    excerpt: string;
  }>;
  model_name?: string;
}

export interface SSEThinkingEvent {
  type: "thinking";
  token: string;
}

export interface SSETokenEvent {
  type: "token";
  token: string;
}

