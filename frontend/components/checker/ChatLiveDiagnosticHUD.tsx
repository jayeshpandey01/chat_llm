"use client";

import React, { useState } from "react";
import { SSEMetadataEvent } from "@/lib/types";
import { Activity, ShieldAlert, AlertTriangle, BookOpen, ChevronDown, ChevronUp, Check, ExternalLink } from "lucide-react";

interface ChatLiveDiagnosticHUDProps {
  meta: SSEMetadataEvent | null;
}

export default function ChatLiveDiagnosticHUD({ meta }: ChatLiveDiagnosticHUDProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!meta) return null;

  const isEmergency = meta.is_emergency;
  const isAcuteOnChronic = meta.is_acute_on_chronic;

  return (
    <div className="border-2 border-black bg-white text-black shadow-[3px_3px_0px_#000] mb-3">
      {/* Top Banner Bar */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className={`p-2.5 flex items-center justify-between cursor-pointer select-none transition-all ${
          isEmergency
            ? "bg-red-600 text-white"
            : isAcuteOnChronic
            ? "bg-amber-400 text-black"
            : "bg-[var(--accent-color)] text-[var(--accent-text)]"
        }`}
      >
        <div className="flex items-center gap-2">
          {isEmergency ? (
            <ShieldAlert className="w-4 h-4 stroke-[2.5]" />
          ) : (
            <Activity className="w-4 h-4 stroke-[2.5]" />
          )}
          <span className="font-['Space_Grotesk'] text-xs font-bold uppercase tracking-tight">
            Live Clinical Diagnostic HUD • {meta.member_name} ({meta.member_relationship})
          </span>
          <span className="neo-badge neo-badge-green text-[9px]">
            {meta.top_condition ? `${meta.top_confidence?.toFixed(1)}% Calibrated` : "Diagnostic Active"}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono opacity-80 hidden sm:inline">
            {isExpanded ? "Collapse HUD" : "Expand HUD"}
          </span>
          <button type="button" className="p-0.5 hover:opacity-75">
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="p-3 bg-neutral-50/70 border-t border-black space-y-2 text-xs">
          {/* Key Findings Row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {/* Top Condition */}
            <div className="p-2 border border-black bg-white shadow-[1px_1px_0px_#000]">
              <div className="text-[9px] font-['Space_Grotesk'] font-bold uppercase opacity-75">
                Primary Assessment:
              </div>
              <div className="font-['Space_Grotesk'] font-extrabold text-xs uppercase text-amber-950 truncate">
                {meta.top_condition || "Synthesizing presentation..."}
              </div>
              <div className="text-[10px] font-mono text-emerald-700 font-bold">
                Confidence: {meta.top_confidence ? `${meta.top_confidence.toFixed(1)}%` : "N/A"}
              </div>
            </div>

            {/* Triage Level */}
            <div className="p-2 border border-black bg-white shadow-[1px_1px_0px_#000]">
              <div className="text-[9px] font-['Space_Grotesk'] font-bold uppercase opacity-75">
                ESI Triage Level:
              </div>
              <div className="font-['Space_Grotesk'] font-bold text-xs uppercase text-blue-900 truncate">
                {meta.triage_level || "Routine Outpatient"}
              </div>
              <div className="text-[10px] opacity-75 truncate">
                {meta.triage_recommendation || "Continue clinical inquiry"}
              </div>
            </div>

            {/* Conformal Guarantee */}
            <div className="p-2 border border-black bg-white shadow-[1px_1px_0px_#000]">
              <div className="text-[9px] font-['Space_Grotesk'] font-bold uppercase opacity-75">
                95% Conformal Set:
              </div>
              <div className="text-[10px] font-mono font-semibold truncate">
                {meta.conformal_set && meta.conformal_set.length > 0
                  ? meta.conformal_set.slice(0, 2).join(", ")
                  : "Calibrating..."}
              </div>
              <div className="text-[9px] font-mono text-neutral-500">
                Finite-sample coverage guaranteed
              </div>
            </div>
          </div>

          {/* Emergency / Acute alerts */}
          {isEmergency && (
            <div className="p-2 border-2 border-red-600 bg-red-100 text-red-950 font-bold flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-700 shrink-0" />
              <span>High Acuity Alert: Immediate in-person medical evaluation strongly indicated.</span>
            </div>
          )}

          {isAcuteOnChronic && meta.acute_on_chronic_alert && (
            <div className="p-2 border-2 border-amber-600 bg-amber-100 text-amber-950 font-bold flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
              <span>Acute-on-Chronic Sentinel: {meta.acute_on_chronic_alert}</span>
            </div>
          )}

          {/* Retrieved Evidence Citations (NIH MedlinePlus / PubMed) */}
          {meta.retrieved_evidence && meta.retrieved_evidence.length > 0 && (
            <div className="pt-1.5 border-t border-neutral-300">
              <div className="text-[10px] font-['Space_Grotesk'] font-bold uppercase text-neutral-700 mb-1 flex items-center gap-1">
                <BookOpen className="w-3 h-3 text-blue-700" />
                NIH MedlinePlus / Peer-Reviewed Literature Attributions:
              </div>
              <div className="flex gap-1.5 flex-wrap">
                {meta.retrieved_evidence.slice(0, 3).map((item, eIdx) => (
                  <span
                    key={eIdx}
                    className="inline-flex items-center gap-1 px-2 py-0.5 border border-neutral-400 bg-white text-[10px] font-mono"
                    title={item.excerpt}
                  >
                    <span className="font-bold">{item.source_title || item.condition}</span>
                    <span className="opacity-60 text-[9px]">({item.source_type})</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

