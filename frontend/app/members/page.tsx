"use client";

import { useEffect, useState } from "react";
import { addMemberCondition, createMember, listMembers } from "@/lib/api";
import { MemberProfile } from "@/lib/types";
import { useAuth } from "@/lib/auth-context";
import { X } from "lucide-react";

export default function FamilyVaultPage() {
  const { user, demoUser, loading } = useAuth();
  const [members, setMembers] = useState<MemberProfile[]>([]);
  const [selectedMember, setSelectedMember] = useState<MemberProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // New Member Modal State
  const [isCreating, setIsCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newRelationship, setNewRelationship] = useState("OTHER");
  const [newAge, setNewAge] = useState<number>(40);
  const [newSex, setNewSex] = useState("MALE");
  const [newChronic, setNewChronic] = useState("");

  // New Condition State
  const [newConditionName, setNewConditionName] = useState("");
  const [newConditionDate, setNewConditionDate] = useState("");

  const loadMembers = async () => {
    try {
      setIsLoading(true);
      setLoadError(null);
      const res = await listMembers();
      setMembers(res.members || []);
      if (res.members && res.members.length > 0) {
        setSelectedMember(res.members[0]);
      }
    } catch (err: any) {
      console.error("Failed to load members:", err);
      setLoadError(err?.message || "Failed to load profiles from database.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!loading) {
      loadMembers();
    }
  }, [user?.uid, demoUser?.uid, loading]);

  const handleCreateMember = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;

    try {
      const chronicList = newChronic
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);

      const res = await createMember({
        name: newName.trim(),
        relationship: newRelationship,
        age: newAge,
        biological_sex: newSex,
        chronic_conditions: chronicList,
        current_medications: [],
        drug_allergies: [],
      });

      await loadMembers();
      setSelectedMember(res.member);
      setIsCreating(false);
      setNewName("");
      setNewChronic("");
    } catch (err: any) {
      alert(`Error creating member: ${err.message}`);
    }
  };

  const handleAddCondition = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newConditionName.trim() || !selectedMember) return;

    try {
      const res = await addMemberCondition(selectedMember.member_id, {
        condition: newConditionName.trim(),
        diagnosed_date: newConditionDate || new Date().toISOString().split("T")[0],
        status: "CHRONIC_ACTIVE",
      });

      setSelectedMember(res.member);
      setMembers((prev) =>
        prev.map((m) => (m.member_id === res.member.member_id ? res.member : m))
      );
      setNewConditionName("");
      setNewConditionDate("");
    } catch (err: any) {
      alert(`Error adding condition: ${err.message}`);
    }
  };

  return (
    <div className="p-6 min-h-[640px]">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b-2 border-[var(--border-color)] mb-6 gap-4">
        <div>
          <h2 className="font-['Space_Grotesk'] text-2xl font-extrabold uppercase">
            Multi-Patient Family Vault
          </h2>
          <p className="text-xs opacity-75">
            Scoped under <code>users/&#123;uid&#125;/members/</code> to maintain zero data contamination between patient profiles.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setIsCreating(true)}
          className="neo-btn neo-btn-primary"
        >
          + Register Family Member
        </button>
      </div>

      {loadError && (
        <div className="p-3 mb-6 border-2 border-red-500 bg-red-100 text-red-900 text-xs flex items-center justify-between gap-3 shadow-[2px_2px_0px_rgba(239,68,68,1)]">
          <div>
            <span className="font-bold text-red-700 mr-2">⚠️ Database Notice:</span>
            <span>{loadError}</span>
          </div>
          <button
            type="button"
            onClick={loadMembers}
            className="neo-btn !py-1 !px-2.5 !text-[11px] bg-white text-black hover:bg-neutral-100 shrink-0 font-bold"
          >
            Retry
          </button>
        </div>
      )}

      {isLoading ? (
        <div className="p-12 text-center text-xs opacity-50 font-mono">Loading family vault profiles...</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Member Card Roster (4 cols) */}
          <div className="lg:col-span-4 space-y-3">
            <h3 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-2">Registered Profiles</h3>
            {members.map((member) => {
              const isSelected = selectedMember?.member_id === member.member_id;
              return (
                <div
                  key={member.member_id}
                  onClick={() => setSelectedMember(member)}
                  className={`p-4 border-2 border-[var(--border-color)] cursor-pointer transition-all ${
                    isSelected
                      ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[4px_4px_0px_var(--shadow-color)] -translate-y-0.5"
                      : "bg-white text-black hover:bg-neutral-50"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="neo-badge neo-badge-yellow text-[10px]">
                      {member.relationship}
                    </span>
                    <span className="text-[10px] font-mono opacity-75">{member.member_id}</span>
                  </div>

                  <div className="font-['Space_Grotesk'] text-base font-extrabold uppercase">
                    {member.name} ({member.age || "?"}y {member.biological_sex})
                  </div>

                  <div className="text-xs opacity-80 mt-1 line-clamp-1">
                    {member.chronic_conditions.length > 0
                      ? `Chronic: ${member.chronic_conditions.join(", ")}`
                      : "No chronic conditions"}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Member Detail & Longitudinal Timeline (8 cols) */}
          <div className="lg:col-span-8 border-2 border-[var(--border-color)] p-6 bg-white text-black shadow-[4px_4px_0px_var(--shadow-color)]">
            {selectedMember ? (
              <div>
                <div className="flex items-center justify-between pb-3 border-b-2 border-black mb-4">
                  <div>
                    <h3 className="font-['Space_Grotesk'] text-2xl font-extrabold uppercase">
                      {selectedMember.name}
                    </h3>
                    <p className="text-xs font-mono opacity-75">
                      Relationship: {selectedMember.relationship} • {selectedMember.age} Years Old • {selectedMember.biological_sex}
                    </p>
                  </div>
                  <span className="neo-badge neo-badge-green font-bold">Active Patient</span>
                </div>

                {/* Clinical Baselines Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
                  <div className="p-3 border border-black bg-neutral-50 text-xs">
                    <div className="font-bold uppercase font-['Space_Grotesk'] mb-1">Chronic Conditions:</div>
                    <div className="opacity-90">
                      {selectedMember.chronic_conditions.length > 0
                        ? selectedMember.chronic_conditions.join(", ")
                        : "None reported"}
                    </div>
                  </div>

                  <div className="p-3 border border-black bg-neutral-50 text-xs">
                    <div className="font-bold uppercase font-['Space_Grotesk'] mb-1">Current Medications:</div>
                    <div className="opacity-90">
                      {selectedMember.current_medications.length > 0
                        ? selectedMember.current_medications.join(", ")
                        : "None reported"}
                    </div>
                  </div>

                  <div className="p-3 border border-black bg-neutral-50 text-xs">
                    <div className="font-bold uppercase font-['Space_Grotesk'] mb-1">Drug Allergies:</div>
                    <div className="opacity-90 font-bold text-red-700">
                      {selectedMember.drug_allergies.length > 0
                        ? selectedMember.drug_allergies.join(", ")
                        : "No known allergies"}
                    </div>
                  </div>
                </div>

                {/* Longitudinal Timeline */}
                <div className="mb-6">
                  <div className="flex items-center justify-between pb-2 border-b border-black mb-3">
                    <h4 className="font-['Space_Grotesk'] text-sm font-bold uppercase">
                      Longitudinal Disease Timeline & Past Encounters
                    </h4>
                  </div>

                  {selectedMember.longitudinal_summary.length === 0 ? (
                    <div className="p-4 border border-dashed border-black/40 text-xs text-center opacity-60">
                      No past clinical episodes recorded for {selectedMember.name}.
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {selectedMember.longitudinal_summary.map((entry, idx) => (
                        <div
                          key={idx}
                          className="p-3 border border-black bg-neutral-50 flex items-center justify-between text-xs font-mono"
                        >
                          <div>
                            <span className="font-bold text-black font-['Space_Grotesk'] uppercase text-sm mr-2">
                              {entry.condition}
                            </span>
                            <span className="opacity-60">{entry.diagnosed_date}</span>
                          </div>
                          <span className="neo-badge neo-badge-yellow text-[10px]">
                            {entry.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Add Diagnosis Milestone Form */}
                <form onSubmit={handleAddCondition} className="p-4 border-2 border-dashed border-black bg-neutral-50">
                  <div className="font-['Space_Grotesk'] text-xs font-bold uppercase mb-2">
                    Append Diagnosed Condition / Milestone to Timeline:
                  </div>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="text"
                      placeholder="Condition name (e.g., Coronary Artery Disease)..."
                      value={newConditionName}
                      onChange={(e) => setNewConditionName(e.target.value)}
                      className="flex-1 p-2 border border-black text-xs font-['Inter']"
                    />
                    <input
                      type="date"
                      value={newConditionDate}
                      onChange={(e) => setNewConditionDate(e.target.value)}
                      className="p-2 border border-black text-xs font-mono"
                    />
                    <button type="submit" className="neo-btn !py-2 !text-xs bg-black text-white hover:bg-neutral-800">
                      + Add Record
                    </button>
                  </div>
                </form>

              </div>
            ) : (
              <div className="text-center p-12 text-xs opacity-50">Select a family member profile from the roster.</div>
            )}
          </div>

        </div>
      )}

      {/* Register Member Modal */}
      {isCreating && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
          <div className="border-3 border-black bg-[#f3f0e8] text-black max-w-md w-full p-6 shadow-[8px_8px_0px_#000]">
            <div className="flex items-center justify-between pb-3 border-b-2 border-black mb-4">
              <h3 className="font-['Space_Grotesk'] text-lg font-bold uppercase">Register Family Member</h3>
              <button type="button" onClick={() => setIsCreating(false)} className="hover:opacity-75"><X className="w-5 h-5" /></button>
            </div>

            <form onSubmit={handleCreateMember} className="space-y-3 text-xs">
              <div>
                <label className="font-bold uppercase block mb-1">Full Name / Label:</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Dad, Mom, Jane Doe"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full p-2 border-2 border-black bg-white"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="font-bold uppercase block mb-1">Relationship:</label>
                  <select
                    value={newRelationship}
                    onChange={(e) => setNewRelationship(e.target.value)}
                    className="w-full p-2 border-2 border-black bg-white uppercase font-bold text-xs"
                  >
                    <option value="FATHER">Father</option>
                    <option value="MOTHER">Mother</option>
                    <option value="SPOUSE">Spouse</option>
                    <option value="CHILD">Child</option>
                    <option value="OTHER">Other</option>
                  </select>
                </div>

                <div>
                  <label className="font-bold uppercase block mb-1">Age:</label>
                  <input
                    type="number"
                    min="0"
                    max="125"
                    value={newAge}
                    onChange={(e) => setNewAge(parseInt(e.target.value) || 0)}
                    className="w-full p-2 border-2 border-black bg-white font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="font-bold uppercase block mb-1">Biological Sex:</label>
                <select
                  value={newSex}
                  onChange={(e) => setNewSex(e.target.value)}
                  className="w-full p-2 border-2 border-black bg-white uppercase font-bold text-xs"
                >
                  <option value="MALE">Male</option>
                  <option value="FEMALE">Female</option>
                  <option value="INTERSEX">Intersex</option>
                </select>
              </div>

              <div>
                <label className="font-bold uppercase block mb-1">Known Chronic Conditions (Comma separated):</label>
                <input
                  type="text"
                  placeholder="e.g. Hypertension, Diabetes, Asthma"
                  value={newChronic}
                  onChange={(e) => setNewChronic(e.target.value)}
                  className="w-full p-2 border-2 border-black bg-white"
                />
              </div>

              <div className="flex gap-2 pt-3">
                <button
                  type="button"
                  onClick={() => setIsCreating(false)}
                  className="neo-btn flex-1 justify-center bg-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="neo-btn neo-btn-primary flex-1 justify-center"
                >
                  Save Profile
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}

