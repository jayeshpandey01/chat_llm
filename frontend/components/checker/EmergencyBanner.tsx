"use client";

import { EmergencyAlert } from "@/lib/types";
import { AlertTriangle, PhoneCall } from "lucide-react";

interface EmergencyBannerProps {
  alert?: EmergencyAlert | null;
}

export default function EmergencyBanner({ alert }: EmergencyBannerProps) {
  if (!alert || !alert.is_emergency) return null;

  return (
    <div className="p-4 border-2 border-[var(--accent-red)] bg-[#fee2e2] text-[#991b1b] pulse-emergency mb-4 shadow-[4px_4px_0px_var(--shadow-color)]">
      <div className="font-['Space_Grotesk'] font-extrabold uppercase text-sm flex items-center justify-between">
        <span className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 stroke-[2.5]" />
          <span>URGENT EMERGENCY ALERT (ESI LEVEL {alert.acuity_level})</span>
        </span>
        <span className="neo-badge neo-badge-red text-[10px]">Action Required: {alert.action}</span>
      </div>
      <p className="text-xs mt-2 font-semibold leading-relaxed font-['Inter']">
        {alert.message}
      </p>
      <div className="mt-3 pt-2 border-t border-[#f87171] flex items-center justify-between text-[11px] font-mono">
        <span>Suspected Critical Life Threat: <b>{alert.condition}</b></span>
        <span className="underline cursor-pointer font-bold flex items-center gap-1">
          <PhoneCall className="w-3.5 h-3.5" /> Call Emergency Dispatch (911 / 112) Immediately
        </span>
      </div>
    </div>
  );
}
