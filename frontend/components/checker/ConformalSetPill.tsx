"use client";

interface ConformalSetPillProps {
  conformalSet: string[];
  coverageGuaranteed: boolean;
}

export default function ConformalSetPill({ conformalSet, coverageGuaranteed }: ConformalSetPillProps) {
  if (!conformalSet || conformalSet.length === 0) return null;

  return (
    <div className="border-2 border-[var(--border-color)] p-3 bg-[var(--card-bg)] mb-4 shadow-[3px_3px_0px_var(--shadow-color)]">
      <div className="flex items-center justify-between mb-2">
        <span className="font-['Space_Grotesk'] text-xs font-bold uppercase">
          Adaptive Conformal Differential Set:
        </span>
        <span className="neo-badge neo-badge-yellow text-[10px]">
          {coverageGuaranteed ? "95% Coverage Bound" : "Empirical Bound"}
        </span>
      </div>

      <div className="space-y-1.5">
        {conformalSet.map((cond, idx) => (
          <div
            key={cond}
            className="flex items-center justify-between p-2 border border-[var(--border-color)] bg-white text-black text-xs font-medium"
          >
            <span className="font-bold">
              {idx + 1}. {cond}
            </span>
            <span className="font-mono text-[11px] opacity-75">
              Included in 95% Set
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

