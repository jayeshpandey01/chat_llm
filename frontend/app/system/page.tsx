"use client";

import { useEffect, useState } from "react";
import { getSystemHealth, getSystemStatus, rollbackModel, triggerContinualLearning } from "@/lib/api";

export default function ContinualOpsPage() {
  const [health, setHealth] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const loadOpsData = async () => {
    try {
      setIsLoading(true);
      const [h, s] = await Promise.all([getSystemHealth(), getSystemStatus()]);
      setHealth(h);
      setStatus(s);
    } catch (err: any) {
      console.warn("Failed to fetch ops telemetry:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadOpsData();
  }, []);

  const handleTriggerLearning = async () => {
    try {
      setActionMessage("Submitting background LoRA alignment training job...");
      const res = await triggerContinualLearning(true);
      setActionMessage(`Continual learning cycle queued! Job ID: ${res.job_id || "Active"}`);
      loadOpsData();
    } catch (err: any) {
      setActionMessage(`Error triggering learning: ${err.message}`);
    }
  };

  const handleRollback = async () => {
    try {
      setActionMessage("Executing atomic rollback to golden anchor...");
      const res = await rollbackModel();
      setActionMessage(`Model rolled back to Version ${res.model_version}! Anchor restored.`);
      loadOpsData();
    } catch (err: any) {
      setActionMessage(`Error during rollback: ${err.message}`);
    }
  };

  return (
    <div className="p-6 min-h-[640px]">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b-2 border-[var(--border-color)] mb-6 gap-4">
        <div>
          <h2 className="font-['Space_Grotesk'] text-2xl font-extrabold uppercase">
            Autonomous Continual Learning Ops
          </h2>
          <p className="text-xs opacity-75">
            Monitor experience replay buffer distributions, release gate hypotheses, and zero-downtime model rollbacks.
          </p>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleTriggerLearning}
            className="neo-btn neo-btn-primary"
          >
            Force Learning Cycle
          </button>
          <button
            type="button"
            onClick={handleRollback}
            className="neo-btn neo-btn-danger"
          >
            Rollback to Anchor
          </button>
        </div>
      </div>

      {actionMessage && (
        <div className="p-3 bg-neutral-100 border-2 border-black text-xs font-mono font-bold mb-6">
          {actionMessage}
        </div>
      )}

      {isLoading ? (
        <div className="p-12 text-center text-xs opacity-50 font-mono">Loading serving telemetry...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Card 1: Active Serving Manager */}
          <div className="border-2 border-[var(--border-color)] p-5 bg-white text-black shadow-[4px_4px_0px_var(--shadow-color)]">
            <div className="flex items-center justify-between pb-2 border-b-2 border-black mb-4">
              <h3 className="font-['Space_Grotesk'] text-lg font-bold uppercase">Active Serving State</h3>
              <span className="neo-badge neo-badge-green">
                Version {health?.model_version ?? 1}
              </span>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="flex justify-between border-b pb-1.5">
                <span className="opacity-75">Active Model Tag:</span>
                <span className="font-bold">{status?.active_model || "qwen2.5:0.5b"}</span>
              </div>
              <div className="flex justify-between border-b pb-1.5">
                <span className="opacity-75">Gateway Status:</span>
                <span className="font-bold text-emerald-700">{health?.status || "HEALTHY"}</span>
              </div>
              <div className="flex justify-between border-b pb-1.5">
                <span className="opacity-75">Active Readers (RWLock):</span>
                <span className="font-bold">{status?.serving_manager?.active_readers ?? 0}</span>
              </div>
              <div className="flex justify-between pb-1">
                <span className="opacity-75">Active Background Job:</span>
                <span className="font-bold">{status?.learning_job_id || "None (Idle)"}</span>
              </div>
            </div>
          </div>

          {/* Card 2: Replay Buffer Distribution */}
          <div className="border-2 border-[var(--border-color)] p-5 bg-white text-black shadow-[4px_4px_0px_var(--shadow-color)]">
            <div className="flex items-center justify-between pb-2 border-b-2 border-black mb-4">
              <h3 className="font-['Space_Grotesk'] text-lg font-bold uppercase">Experience Replay Buffer</h3>
              <span className="neo-badge neo-badge-yellow">70/30 Mixture</span>
            </div>

            <p className="text-xs opacity-75 mb-4 font-['Inter']">
              Protects against catastrophic forgetting by preserving anchor baseline distributions alongside new web curation.
            </p>

            <div className="grid grid-cols-2 gap-3 font-mono">
              <div className="p-3 border border-black bg-emerald-50">
                <div className="text-[10px] font-bold uppercase text-emerald-900">Anchor Ratio</div>
                <div className="text-2xl font-bold">70.0%</div>
                <div className="text-[10px] opacity-75 mt-1">Non-negotiable Floor</div>
              </div>
              <div className="p-3 border border-black bg-blue-50">
                <div className="text-[10px] font-bold uppercase text-blue-900">Web Stream Ratio</div>
                <div className="text-2xl font-bold">30.0%</div>
                <div className="text-[10px] opacity-75 mt-1">Curated Telemetry</div>
              </div>
            </div>
          </div>

          {/* Card 3: Four-Bucket Statistical Release Gates */}
          <div className="md:col-span-2 border-2 border-[var(--border-color)] p-5 bg-white text-black shadow-[4px_4px_0px_var(--shadow-color)]">
            <div className="flex items-center justify-between pb-2 border-b-2 border-black mb-4">
              <h3 className="font-['Space_Grotesk'] text-lg font-bold uppercase">
                Statistical Release Gating Criteria
              </h3>
              <span className="neo-badge text-[10px]">Non-Parametric</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div className="p-3 border border-black bg-neutral-50">
                <div className="font-bold uppercase mb-1">1. Paired Bootstrap CI:</div>
                <div className="text-emerald-700 font-bold">Δ &gt; 0.0 at 95% Confidence</div>
                <p className="text-[11px] opacity-70 mt-1 font-['Inter']">1000 resamples over stratified benchmark suite.</p>
              </div>

              <div className="p-3 border border-black bg-neutral-50">
                <div className="font-bold uppercase mb-1">2. Wilcoxon Signed-Rank:</div>
                <div className="text-emerald-700 font-bold">p &lt; 0.05 Significance</div>
                <p className="text-[11px] opacity-70 mt-1 font-['Inter']">Inter-candidate pairwise score comparison.</p>
              </div>

              <div className="p-3 border border-black bg-neutral-50">
                <div className="font-bold uppercase mb-1">3. McNemar Continuity:</div>
                <div className="text-emerald-700 font-bold">Zero Degradation Gating</div>
                <p className="text-[11px] opacity-70 mt-1 font-['Inter']">Ensures candidate never forgets anchor skills.</p>
              </div>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

