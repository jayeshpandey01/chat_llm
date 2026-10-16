"use client";

import { X, Check } from "lucide-react";

export interface AlgorithmDetail {
  id: string;
  name: string;
  scope: string;
  algorithm: string;
  dataset: string;
  trainSamples: string;
  testSamples: string;
  trainAccuracy: string;
  testTop1: string;
  testTop3: string;
  testTop5: string;
  macroF1: string;
  calibration: string;
  checkpoint: string;
  fitTime: string;
  pipelineDescription: string;
  keyInnovations: string[];
  sentinelsHandled: string[];
}

interface AlgorithmPipelineModalProps {
  algorithm: AlgorithmDetail | null;
  onClose: () => void;
}

export default function AlgorithmPipelineModal({ algorithm, onClose }: AlgorithmPipelineModalProps) {
  if (!algorithm) return null;

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="border-3 border-[var(--border-color)] bg-[var(--bg-color)] text-[var(--text-color)] max-w-2xl w-full p-6 shadow-[8px_8px_0px_var(--shadow-color)] max-h-[85vh] overflow-y-auto">
        
        {/* Header */}
        <div className="flex items-start justify-between pb-3 border-b-2 border-[var(--border-color)] mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="neo-badge neo-badge-green font-mono">{algorithm.scope}</span>
              <span className="text-[10px] font-mono opacity-70">Artifact: {algorithm.checkpoint}</span>
            </div>
            <h3 className="font-['Space_Grotesk'] text-2xl font-extrabold uppercase">
              {algorithm.name}
            </h3>
            <p className="text-xs font-mono opacity-80 mt-0.5">
              Algorithm: <b>{algorithm.algorithm}</b> • Fit Time: {algorithm.fitTime}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="hover:opacity-70 p-1"
            title="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Metrics Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4 font-mono text-xs">
          <div className="p-2.5 border-2 border-[var(--border-color)] bg-white text-black">
            <span className="text-[10px] uppercase opacity-75 block font-bold">Train Accuracy</span>
            <span className="text-xl font-bold font-['Space_Grotesk']">{algorithm.trainAccuracy}</span>
            <span className="text-[9px] opacity-60 block mt-0.5">{algorithm.trainSamples}</span>
          </div>

          <div className="p-2.5 border-2 border-[var(--border-color)] bg-[var(--accent-color)] text-[var(--accent-text)]">
            <span className="text-[10px] uppercase opacity-75 block font-bold">Test Top-1</span>
            <span className="text-xl font-bold font-['Space_Grotesk']">{algorithm.testTop1}</span>
            <span className="text-[9px] opacity-75 block mt-0.5">{algorithm.testSamples}</span>
          </div>

          <div className="p-2.5 border-2 border-[var(--border-color)] bg-white text-black">
            <span className="text-[10px] uppercase opacity-75 block font-bold">Test Top-3 / Top-5</span>
            <span className="text-base font-bold font-['Space_Grotesk']">{algorithm.testTop3} / {algorithm.testTop5}</span>
            <span className="text-[9px] opacity-60 block mt-0.5">Differential Coverage</span>
          </div>

          <div className="p-2.5 border-2 border-[var(--border-color)] bg-white text-black">
            <span className="text-[10px] uppercase opacity-75 block font-bold">Macro F1 / Calibration</span>
            <span className="text-base font-bold font-['Space_Grotesk']">{algorithm.macroF1}</span>
            <span className="text-[9px] opacity-60 block mt-0.5">{algorithm.calibration}</span>
          </div>
        </div>

        {/* Pipeline Summary in Short and Simple */}
        <div className="space-y-4 text-xs font-['Inter'] mb-6">
          <div>
            <h4 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-1">
              Training Pipeline &amp; Data Handling:
            </h4>
            <p className="leading-relaxed bg-white text-black p-3 border-2 border-[var(--border-color)]">
              {algorithm.pipelineDescription}
            </p>
          </div>

          <div>
            <h4 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-1.5">
              Engineering Architecture &amp; Key Design Rationale:
            </h4>
            <ul className="space-y-1.5 list-disc pl-4 opacity-90 leading-normal">
              {algorithm.keyInnovations.map((item, idx) => (
                <li key={idx}><b>{item.split(":")[0]}:</b>{item.split(":")[1] || ""}</li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-1.5">
              Integrated Sentinels &amp; Clinical Safeguards:
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {algorithm.sentinelsHandled.map((s, idx) => (
                <span key={idx} className="neo-badge neo-badge-green text-[10px] inline-flex items-center">
                  <Check className="w-3 h-3 mr-1" />
                  {s}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-3 border-t-2 border-[var(--border-color)] flex items-center justify-between">
          <span className="text-[11px] font-mono opacity-70">
            Validated across 265/265 automated test suites
          </span>
          <button
            type="button"
            onClick={onClose}
            className="neo-btn neo-btn-primary !py-2 !px-5"
          >
            Close Details
          </button>
        </div>

      </div>
    </div>
  );
}

