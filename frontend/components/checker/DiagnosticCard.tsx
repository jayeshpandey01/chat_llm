"use client";

import { ConditionExplanation } from "@/lib/types";
import { Check, Circle, ExternalLink } from "lucide-react";

interface DiagnosticCardProps {
  explanation: ConditionExplanation;
  onOpenGuideline?: (condition: string, topic?: string | null, summary?: string | null) => void;
}

export default function DiagnosticCard({ explanation, onOpenGuideline }: DiagnosticCardProps) {
  const isTopRank = explanation.rank === 1;

  return (
    <div className={`border-2 border-[var(--border-color)] p-4 bg-[var(--card-bg)] shadow-[3px_3px_0px_var(--shadow-color)] transition-all ${
      isTopRank ? "mb-4 border-l-8 border-l-[var(--border-color)]" : "mb-3"
    }`}>
      {/* Header Row */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono font-bold uppercase opacity-75">
          {isTopRank ? "Primary Candidate (Rank 1)" : `Differential Rank #${explanation.rank}`}
        </span>
        <span className="text-[11px] font-mono opacity-80">
          Source: {explanation.source_model.includes("41") ? "Primary Care (41)" : "General (50)"}
        </span>
      </div>

      {/* Condition & Confidence Score */}
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="font-['Space_Grotesk'] text-xl font-extrabold uppercase">
          {explanation.condition}
        </h3>
        <span className="font-['Space_Grotesk'] text-lg font-bold text-[var(--accent-blue)]">
          {explanation.confidence_percentage.toFixed(1)}%
        </span>
      </div>

      {/* Confidence Bar */}
      <div className="w-full h-2.5 border-2 border-[var(--border-color)] bg-white mb-3 overflow-hidden">
        <div
          className="h-full bg-[var(--accent-color)] transition-all duration-500"
          style={{ width: `${Math.min(100, Math.max(5, explanation.confidence_percentage))}%` }}
        />
      </div>

      {/* Plain Language Explanation */}
      <p className="text-xs leading-relaxed opacity-90 mb-3">
        {explanation.explanation_text}
      </p>

      {/* Matched Evidence Symptoms Chips */}
      {explanation.matched_symptoms.length > 0 && (
        <div className="mb-3">
          <div className="text-[11px] font-bold uppercase opacity-75 mb-1.5 font-['Space_Grotesk']">
            Matched Supporting Symptoms:
          </div>
          <div className="flex flex-wrap gap-1.5">
            {explanation.matched_symptoms.map((s) => (
              <span key={s} className="neo-badge neo-badge-green text-[10px] flex items-center gap-1">
                <Check className="w-3 h-3 stroke-[3]" /> {s.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Missing Classic Hallmarks */}
      {explanation.missing_classic_symptoms.length > 0 && (
        <div className="mb-3">
          <div className="text-[11px] font-bold uppercase opacity-75 mb-1.5 font-['Space_Grotesk']">
            Classic Hallmarks Not Reported:
          </div>
          <div className="flex flex-wrap gap-1.5">
            {explanation.missing_classic_symptoms.map((s) => (
              <span key={s} className="neo-badge text-[10px] opacity-70 flex items-center gap-1">
                <Circle className="w-2.5 h-2.5 stroke-[2]" /> {s.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Footer: Diagnostic Odds Ratio & NIH Guideline link */}
      <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-[var(--border-color)]">
        <div className="text-[11px] opacity-75">
          {explanation.evidence_symptoms.length > 0 && (
            <span>Top DOR: <b>{explanation.evidence_symptoms[0].diagnostic_odds_ratio.toFixed(1)}x</b></span>
          )}
        </div>

        {explanation.nih_guideline_topic && onOpenGuideline && (
          <button
            type="button"
            onClick={() => onOpenGuideline(explanation.condition, explanation.nih_guideline_topic, explanation.nih_guideline_summary)}
            className="underline font-bold text-xs hover:text-[var(--accent-blue)] inline-flex items-center gap-1"
          >
            NIH MedQuAD Guideline <ExternalLink className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  );
}
