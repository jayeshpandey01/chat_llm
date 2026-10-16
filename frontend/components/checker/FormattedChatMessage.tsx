"use client";

import React from "react";
import { AlertTriangle, AlertCircle, ShieldAlert } from "lucide-react";

interface FormattedChatMessageProps {
  text: string;
  sender: "patient" | "assistant";
  isStreaming?: boolean;
}

export default function FormattedChatMessage({ text, sender, isStreaming }: FormattedChatMessageProps) {
  if (!text && isStreaming) {
    return (
      <div className="flex items-center gap-2 text-neutral-500 font-mono text-xs py-1">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
        <span>Synthesizing calibrated clinical response...</span>
      </div>
    );
  }

  // Split into paragraphs
  const paragraphs = text.split("\n\n");

  return (
    <div className="space-y-2 text-xs leading-relaxed font-['Inter']">
      {paragraphs.map((p, idx) => {
        const trimmed = p.trim();
        if (!trimmed) return null;

        // 1. Critical Emergency Alert block
        if (trimmed.includes("[CRITICAL EMERGENCY ALERT]") || trimmed.includes("URGENT EMERGENCY ALERT")) {
          return (
            <div
              key={idx}
              className="p-2.5 border-2 border-red-600 bg-red-50 text-red-950 font-semibold shadow-[2px_2px_0px_#991b1b] flex items-start gap-2"
            >
              <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <div>
                <div className="text-[10px] font-['Space_Grotesk'] font-bold uppercase tracking-wider text-red-700">
                  Critical Emergency Sentinel Triggered
                </div>
                <div className="mt-0.5 text-xs font-normal">
                  {formatInlineText(trimmed.replace(/\[CRITICAL EMERGENCY ALERT\]:?/i, "").replace(/URGENT EMERGENCY ALERT:?/i, "").trim())}
                </div>
              </div>
            </div>
          );
        }

        // 2. Acute-on-Chronic Sentinel block
        if (trimmed.includes("[ACUTE-ON-CHRONIC SENTINEL]") || trimmed.includes("ACUTE-ON-CHRONIC ALERT")) {
          return (
            <div
              key={idx}
              className="p-2.5 border-2 border-amber-600 bg-amber-50 text-amber-950 font-semibold shadow-[2px_2px_0px_#b45309] flex items-start gap-2"
            >
              <AlertCircle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
              <div>
                <div className="text-[10px] font-['Space_Grotesk'] font-bold uppercase tracking-wider text-amber-800">
                  Longitudinal Acute-on-Chronic Alert
                </div>
                <div className="mt-0.5 text-xs font-normal">
                  {formatInlineText(trimmed.replace(/\[ACUTE-ON-CHRONIC SENTINEL\]:?/i, "").replace(/ACUTE-ON-CHRONIC ALERT:?/i, "").trim())}
                </div>
              </div>
            </div>
          );
        }

        // 3. Bullet points
        if (trimmed.startsWith("•") || trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
          const lines = trimmed.split("\n");
          return (
            <ul key={idx} className="space-y-1 pl-1">
              {lines.map((line, lIdx) => {
                const cleanLine = line.replace(/^[•\-\*]\s*/, "");
                return (
                  <li key={lIdx} className="flex items-start gap-1.5">
                    <span className="text-[var(--accent-blue)] font-bold shrink-0">•</span>
                    <span>{formatInlineText(cleanLine)}</span>
                  </li>
                );
              })}
            </ul>
          );
        }

        // 4. Disclaimer note
        const isDisclaimer =
          trimmed.toLowerCase().includes("healthtrack ai is a clinical") ||
          trimmed.toLowerCase().includes("preliminary decision support") ||
          (trimmed.startsWith("*") && trimmed.endsWith("*") && trimmed.length > 20);

        if (isDisclaimer) {
          const cleanDisclaimer = trimmed.replace(/^\*+|\*+$/g, "").trim();
          return (
            <div key={idx} className="text-[10px] opacity-75 italic font-mono pt-2 mt-1.5 border-t border-neutral-300 leading-relaxed">
              {cleanDisclaimer}
            </div>
          );
        }

        // 5. Normal formatted paragraph
        return (
          <p key={idx} className="whitespace-pre-wrap">
            {formatInlineText(trimmed)}
          </p>
        );
      })}
    </div>
  );
}

/**
 * Replaces **bold text** with <strong> and *italic text* with <em>
 */
function formatInlineText(text: string): React.ReactNode {
  // Strip trailing orphan asterisks if currently streaming
  let clean = text;
  if (clean.endsWith("**") && (clean.match(/\*\*/g) || []).length % 2 === 1) {
    clean = clean.slice(0, -2);
  } else if (clean.endsWith("*") && !clean.endsWith("**") && (clean.match(/(?<!\*)\*(?!\*)/g) || []).length % 2 === 1) {
    clean = clean.slice(0, -1);
  }

  // Regex to match **bold** chunks or *italic* chunks
  const parts = clean.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);

  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
      const inner = part.slice(2, -2);
      return (
        <strong key={index} className="font-extrabold text-inherit">
          {inner}
        </strong>
      );
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length >= 2) {
      const inner = part.slice(1, -1);
      return (
        <em key={index} className="italic text-inherit">
          {inner}
        </em>
      );
    }
    return part;
  });
}

