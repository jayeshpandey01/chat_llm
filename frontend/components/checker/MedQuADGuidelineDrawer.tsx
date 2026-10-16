"use client";

import { X, BookOpen, AlertCircle } from "lucide-react";

interface MedQuADGuidelineDrawerProps {
  isOpen: boolean;
  condition: string;
  topic?: string | null;
  summary?: string | null;
  onClose: () => void;
}

export default function MedQuADGuidelineDrawer({
  isOpen,
  condition,
  topic,
  summary,
  onClose,
}: MedQuADGuidelineDrawerProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="border-3 border-[var(--border-color)] bg-[var(--bg-color)] text-[var(--text-color)] max-w-xl w-full p-6 shadow-[8px_8px_0px_var(--shadow-color)]">
        <div className="flex items-center justify-between pb-3 border-b-2 border-[var(--border-color)] mb-4">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-[var(--accent-blue)]" />
            <div>
              <span className="text-[10px] font-mono uppercase opacity-75">NIH MedlinePlus Evidence</span>
              <h3 className="font-['Space_Grotesk'] text-xl font-bold uppercase">{condition}</h3>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 hover:opacity-70 text-[var(--text-color)]"
            title="Close"
            aria-label="Close"
          >
            <X className="w-5 h-5 stroke-[2.5]" />
          </button>
        </div>

        <div className="text-xs leading-relaxed space-y-3 font-['Inter'] mb-6 max-h-[60vh] overflow-y-auto pr-2">
          {topic && (
            <div>
              <span className="font-bold uppercase text-[11px] opacity-80">Guideline Topic:</span>
              <p className="mt-0.5 font-medium">{topic}</p>
            </div>
          )}

          {summary ? (
            <div>
              <span className="font-bold uppercase text-[11px] opacity-80">Clinical Summary Excerpt:</span>
              <p className="mt-0.5 leading-relaxed bg-white/50 p-3 border border-[var(--border-color)]/20 rounded">
                {summary}
              </p>
            </div>
          ) : (
            <p className="opacity-75 italic">
              No specific NIH summary paragraph was extracted for this condition. Standard primary care diagnostic guidelines apply.
            </p>
          )}

          <div className="p-3 bg-amber-50 text-amber-900 border border-amber-300 text-[11px] flex items-start gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <b>Clinical Guidance Note:</b> Grounded evidence derived from peer-reviewed CDC and MedlinePlus topics indexed in <code>clinical_evidence.db</code>.
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="neo-btn neo-btn-primary w-full justify-center !py-2.5"
        >
          Close Guideline
        </button>
      </div>
    </div>
  );
}
