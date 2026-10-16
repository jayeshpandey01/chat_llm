"use client";

import { ActiveQuestionInquiry } from "@/lib/types";
import { HelpCircle } from "lucide-react";

interface ActiveInquiryModalProps {
  inquiry?: ActiveQuestionInquiry | null;
  onSelectAnswer: (answer: string) => void;
}

export default function ActiveInquiryModal({ inquiry, onSelectAnswer }: ActiveInquiryModalProps) {
  if (!inquiry || !inquiry.is_active_inquiry_needed) return null;

  return (
    <div className="border-2 border-dashed border-[var(--border-color)] p-4 bg-white text-black mb-4 shadow-[3px_3px_0px_var(--shadow-color)]">
      <div className="font-['Space_Grotesk'] text-xs font-extrabold uppercase mb-1 flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <HelpCircle className="w-4 h-4 stroke-[2.2]" />
          <span>Targeted Bayesian Clarification Needed</span>
        </span>
        <span className="text-[10px] bg-black text-white px-1.5 py-0.5 font-mono">
          Solitary Symptom Entropy
        </span>
      </div>

      <p className="text-xs mb-3 opacity-90 leading-relaxed font-['Inter']">
        {inquiry.rationale}
      </p>

      <div className="space-y-2">
        {inquiry.high_yield_questions.map((q, idx) => (
          <div key={idx} className="p-2 border border-black/20 bg-neutral-50 flex items-center justify-between gap-2">
            <span className="text-xs font-medium flex items-center gap-1.5">
              <span className="font-bold opacity-70">Q:</span> {q}
            </span>
            <div className="flex gap-1.5 shrink-0">
              <button
                type="button"
                onClick={() => onSelectAnswer(`Yes: ${q}`)}
                className="neo-btn !py-0.5 !px-2 !text-[11px] bg-black text-white hover:bg-neutral-800"
              >
                Yes
              </button>
              <button
                type="button"
                onClick={() => onSelectAnswer(`No: ${q}`)}
                className="neo-btn !py-0.5 !px-2 !text-[11px] bg-white text-black hover:bg-neutral-100"
              >
                No
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
