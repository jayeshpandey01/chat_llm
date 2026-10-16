"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import {
  addMemberCondition,
  createMember,
  createSession,
  getMember,
  getSystemHealth,
  getSystemStatus,
  listMembers,
  listSessions,
  predictDisease,
  rollbackModel,
  triggerContinualLearning,
  uploadMedicalFile,
} from "@/lib/api";
import {
  DiagnosticReportSchema,
  MemberProfile,
  SessionRecord,
  SSEMetadataEvent,
} from "@/lib/types";
import { streamChatConsultation } from "@/lib/sse";
import EmergencyBanner from "@/components/checker/EmergencyBanner";
import DiagnosticCard from "@/components/checker/DiagnosticCard";
import ConformalSetPill from "@/components/checker/ConformalSetPill";
import ActiveInquiryModal from "@/components/checker/ActiveInquiryModal";
import MedQuADGuidelineDrawer from "@/components/checker/MedQuADGuidelineDrawer";
import ThemeToggle from "@/components/ThemeToggle";
import ClinicalClassesBrowser from "@/components/checker/ClinicalClassesBrowser";
import FormattedChatMessage from "@/components/checker/FormattedChatMessage";
import ChatLiveDiagnosticHUD from "@/components/checker/ChatLiveDiagnosticHUD";
import {
  ClipboardList,
  MessageSquare,
  User,
  Activity,
  Settings,
  AlertCircle,
  Check,
  X,
  Plus,
  Paperclip,
  Printer,
} from "lucide-react";

const CHECKLIST_SYMPTOMS = [
  { id: "headache", label: "Headache" },
  { id: "nausea", label: "Nausea" },
  { id: "fever", label: "Fever" },
  { id: "cough", label: "Cough" },
  { id: "chest_pain", label: "Chest Pain" },
  { id: "fatigue", label: "Fatigue" },
  { id: "skin_rash", label: "Skin Rash" },
  { id: "itching", label: "Itching" },
  { id: "shortness_of_breath", label: "Dyspnea" },
  { id: "joint_pain", label: "Joint Pain" },
];

export default function UserDashboardPage() {
  const { user, demoUser, loading, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<"input" | "chatbot" | "profile" | "summary" | "status">("input");

  // Family Members & Roles
  const [members, setMembers] = useState<MemberProfile[]>([]);
  const [activeMember, setActiveMember] = useState<MemberProfile | null>(null);

  // ── 1. INPUT FORM STATE (POST /api/predict) ──
  const [queryText, setQueryText] = useState("Throbbing unilateral headache with nausea and light sensitivity, denies fever");
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [age, setAge] = useState<number | undefined>(34);
  const [biologicalSex, setBiologicalSex] = useState<string>("FEMALE");
  const [isPregnant, setIsPregnant] = useState<boolean>(false);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [report, setReport] = useState<DiagnosticReportSchema | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [modelTier, setModelTier] = useState<"auto" | "primary_41" | "general_50" | "all_580">("auto");
  const [activeLoadedCondition, setActiveLoadedCondition] = useState<string | null>(null);

  // Guideline Drawer
  const [guidelineModal, setGuidelineModal] = useState<{
    isOpen: boolean;
    condition: string;
    topic?: string | null;
    summary?: string | null;
  }>({ isOpen: false, condition: "" });

  // ── 2. CHATBOT STREAM STATE (POST /api/chat/stream) ──
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [activeSession, setActiveSession] = useState<SessionRecord | null>(null);
  const [chatMessages, setChatMessages] = useState<Array<{ id: string; sender: "patient" | "assistant"; text: string; attachedDoc?: string; isStreaming?: boolean }>>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatStreaming, setIsChatStreaming] = useState(false);
  const [chatMetadata, setChatMetadata] = useState<SSEMetadataEvent | null>(null);
  const [attachedFile, setAttachedFile] = useState<{ id: string; name: string } | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const chatFeedRef = useRef<HTMLDivElement>(null);

  // ── 3. PROFILE / NEW MEMBER FORM (POST /api/members) ──
  const [newMemberName, setNewMemberName] = useState("");
  const [newMemberRel, setNewMemberRel] = useState("OTHER");
  const [newMemberAge, setNewMemberAge] = useState<number>(45);
  const [newMemberSex, setNewMemberSex] = useState("MALE");
  const [newMemberChronic, setNewMemberChronic] = useState("");

  // ── 4. SUMMARY / NEW CONDITION FORM (POST /api/members/{id}/conditions) ──
  const [newConditionName, setNewConditionName] = useState("");
  const [newConditionDate, setNewConditionDate] = useState("");

  // ── 5. STATUS / OPS STATE (GET /health, GET /status) ──
  const [healthData, setHealthData] = useState<any>(null);
  const [statusData, setStatusData] = useState<any>(null);
  const [opsMessage, setOpsMessage] = useState<string | null>(null);
  const [dbError, setDbError] = useState<string | null>(null);

  // Initial Load: Fetch Members and Sessions (Waits for Auth to Resolve)
  useEffect(() => {
    if (loading) return;

    async function init() {
      try {
        setDbError(null);
        const memRes = await listMembers();
        const list = memRes.members || [];
        setMembers(list);

        if (list.length > 0) {
          selectRoleMember(list[0]);
        }

        const sessRes = await listSessions();
        const sessList = sessRes.sessions || [];
        setSessions(sessList);
        if (sessList.length > 0) {
          setActiveSession(sessList[0]);
          mapSessionTurns(sessList[0]);
        } else {
          const created = await createSession({ title: "Primary Consultation", member_id: "mem_primary" });
          setSessions([created.session]);
          setActiveSession(created.session);
        }

        // Fetch telemetry
        const [h, s] = await Promise.all([getSystemHealth(), getSystemStatus()]);
        setHealthData(h);
        setStatusData(s);
      } catch (err: any) {
        console.warn("Dashboard initialization notice:", err);
        setDbError(err?.message || "Could not load clinical database. Backend may be offline or initializing.");
      }
    }
    init();
  }, [user?.uid, demoUser?.uid, loading]);

  // When user switches role / member: auto-fill demographics into the symptom form
  const selectRoleMember = (member: MemberProfile) => {
    setActiveMember(member);
    if (member.age !== null && member.age !== undefined) setAge(member.age);
    if (member.biological_sex) setBiologicalSex(member.biological_sex);
    setIsPregnant(!!member.is_pregnant);
  };

  const mapSessionTurns = (sess: SessionRecord) => {
    const turns = (sess.recent_context_window || []).flatMap((t, i) => [
      { id: `u-${i}`, sender: "patient" as const, text: t.user },
      { id: `a-${i}`, sender: "assistant" as const, text: t.assistant },
    ]);
    setChatMessages(turns);
  };

  // ── PREDICTION HANDLER ──
  const handleLoadFromDirectory = (
    conditionName: string,
    symptoms: string[],
    tier: "primary_41" | "general_50" | "all_580"
  ) => {
    setActiveLoadedCondition(conditionName);
    const symptomStr = symptoms.map((s) => s.replace(/_/g, " ")).join(", ");
    setQueryText(`Patient presents with presentation of ${conditionName}: ${symptomStr}.`);
    const cleanSyms = symptoms.map((s) => s.toLowerCase().replace(/ /g, "_"));
    setSelectedSymptoms((prev) => Array.from(new Set([...prev, ...cleanSyms])));
    setModelTier(tier);
  };

  const handleClearLoadedCondition = () => {
    setActiveLoadedCondition(null);
    setQueryText("");
    setSelectedSymptoms([]);
  };

  const handleRunDiagnosis = async () => {
    if (!queryText.trim() && selectedSymptoms.length === 0) {
      setPredictError("Please provide a symptom description or pick checklist symptoms.");
      return;
    }
    setIsPredicting(true);
    setPredictError(null);

    try {
      const res = await predictDisease({
        query_text: queryText,
        selected_symptoms: selectedSymptoms,
        age: age || null,
        biological_sex: biologicalSex || null,
        is_pregnant: isPregnant,
        top_k: 3,
        conformal_coverage: 0.95,
        model_tier: modelTier,
      });
      setReport(res);
    } catch (err: any) {
      setPredictError(err.message || "Failed to execute prediction pipeline.");
    } finally {
      setIsPredicting(false);
    }
  };

  // ── SESSION MANAGEMENT HANDLERS ──
  const handleCreateNewSession = async () => {
    try {
      const title = activeMember ? `${activeMember.name} Consultation` : "General Consultation";
      const created = await createSession({
        title: `${title} #${sessions.length + 1}`,
        member_id: activeMember?.member_id,
      });
      setSessions((prev) => [created.session, ...prev]);
      setActiveSession(created.session);
      setChatMessages([]);
      setChatMetadata(null);
    } catch (err: any) {
      alert(`Failed to create consultation session: ${err.message}`);
    }
  };

  const handleSelectSession = (sessId: string) => {
    const target = sessions.find((s) => s.session_id === sessId);
    if (target) {
      setActiveSession(target);
      mapSessionTurns(target);
      setChatMetadata(null);
    }
  };

  // ── CHAT SSE STREAM HANDLER ──
  const handleSendChatStream = async (overridePrompt?: string) => {
    const query = (overridePrompt || chatInput).trim();
    if (!query || isChatStreaming) return;

    let currentSession = activeSession;
    if (!currentSession) {
      try {
        const title = activeMember ? `${activeMember.name} Consultation` : "General Consultation";
        const created = await createSession({
          title: `${title} #${sessions.length + 1}`,
          member_id: activeMember?.member_id,
        });
        currentSession = created.session;
        setSessions((prev) => [created.session, ...prev]);
        setActiveSession(created.session);
      } catch (err: any) {
        alert("Failed to initialize consultation session: " + err.message);
        return;
      }
    }

    setChatInput("");
    setIsChatStreaming(true);

    const userMsgId = `u-${Date.now()}`;
    const assistantMsgId = `a-${Date.now()}`;

    setChatMessages((prev) => [
      ...prev,
      { id: userMsgId, sender: "patient", text: query, attachedDoc: attachedFile?.name },
      { id: assistantMsgId, sender: "assistant", text: "", isStreaming: true },
    ]);

    await streamChatConsultation({
      sessionId: currentSession.session_id,
      queryText: query,
      documentId: attachedFile?.id,
      onDiagnosticReport: (meta) => setChatMetadata(meta),
      onToken: (token) => {
        setChatMessages((prev) =>
          prev.map((m) => (m.id === assistantMsgId ? { ...m, text: m.text + token } : m))
        );
        if (chatFeedRef.current) {
          chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
        }
      },
      onDone: () => {
        setIsChatStreaming(false);
        setChatMessages((prev) => {
          const updated = prev.map((m) =>
            m.id === assistantMsgId ? { ...m, isStreaming: false } : m
          );
          if (currentSession) {
            const lastUser = updated.find((m) => m.id === userMsgId)?.text || query;
            const lastAssistant = updated.find((m) => m.id === assistantMsgId)?.text || "";
            const existingTurns = currentSession.recent_context_window || [];
            const newTurn = {
              turn: existingTurns.length + 1,
              user: lastUser,
              assistant: lastAssistant,
              top_condition: chatMetadata?.top_condition,
              timestamp: Date.now(),
            };
            const updatedSess: SessionRecord = {
              ...currentSession,
              recent_context_window: [...existingTurns, newTurn],
            };
            setActiveSession(updatedSess);
            setSessions((prevSess) =>
              prevSess.map((s) => (s.session_id === updatedSess.session_id ? updatedSess : s))
            );
          }
          return updated;
        });
        setAttachedFile(null);
      },
      onError: (err) => {
        setIsChatStreaming(false);
        setChatMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  isStreaming: false,
                  text: m.text
                    ? `${m.text}\n[Stream interrupted: ${err.message}]`
                    : `[Clinical service notice: ${err.message}]`,
                }
              : m
          )
        );
      },
    });
  };

  // ── FILE UPLOAD HANDLER ──
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const res = await uploadMedicalFile(file, file.type.includes("pdf") ? "LAB_REPORT" : "CLINICAL_PHOTO");
      setAttachedFile({ id: res.file.file_id, name: res.file.filename });
      alert(`Uploaded ${res.file.filename}. Clinical vision summary extracted.`);
    } catch (err: any) {
      alert(`Upload error: ${err.message}`);
    }
  };

  // ── ADD MEMBER HANDLER ──
  const handleCreateMember = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMemberName.trim()) return;

    try {
      const chronicArr = newMemberChronic.split(",").map((s) => s.trim()).filter(Boolean);
      const res = await createMember({
        name: newMemberName.trim(),
        relationship: newMemberRel,
        age: newMemberAge,
        biological_sex: newMemberSex,
        chronic_conditions: chronicArr,
      });

      // Synchronize full list from DB
      const refreshed = await listMembers();
      const updatedList = refreshed.members && refreshed.members.length > 0 ? refreshed.members : [...members, res.member];
      setMembers(updatedList);
      selectRoleMember(res.member);
      setNewMemberName("");
      setNewMemberChronic("");
      alert(`Registered profile for ${res.member.name}`);
    } catch (err: any) {
      alert(`Error creating member: ${err.message}`);
    }
  };

  // ── ADD MILESTONE HANDLER ──
  const handleAddCondition = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newConditionName.trim() || !activeMember) return;

    try {
      const res = await addMemberCondition(activeMember.member_id, {
        condition: newConditionName.trim(),
        diagnosed_date: newConditionDate || new Date().toISOString().split("T")[0],
        status: "CHRONIC_ACTIVE",
      });

      setActiveMember(res.member);
      setMembers((prev) => prev.map((m) => (m.member_id === res.member.member_id ? res.member : m)));
      setNewConditionName("");
      setNewConditionDate("");
      alert(`Appended milestone to ${res.member.name}'s longitudinal health record.`);
    } catch (err: any) {
      alert(`Error appending condition: ${err.message}`);
    }
  };

  // ── CONTINUAL OPS HANDLERS ──
  const handleTriggerLearning = async () => {
    try {
      setOpsMessage("Submitting background LoRA SFT worker on replay buffer...");
      const res = await triggerContinualLearning(true);
      setOpsMessage(`Continual learning job queued! Job ID: ${res.job_id || "Active"}`);
    } catch (err: any) {
      setOpsMessage(`Error triggering learning: ${err.message}`);
    }
  };

  const handleRollback = async () => {
    try {
      setOpsMessage("Executing atomic rollback to golden anchor...");
      const res = await rollbackModel();
      setOpsMessage(`Rolled back to Model Version ${res.model_version}! Golden anchor restored.`);
    } catch (err: any) {
      setOpsMessage(`Error during rollback: ${err.message}`);
    }
  };

  // Quick NegEx Token Extraction Preview
  const parseTokens = () => {
    const q = queryText.toLowerCase();
    const affirmed: string[] = [];
    const denied: string[] = [];

    if (q.includes("denies fever") || q.includes("no fever")) denied.push("Fever");
    else if (q.includes("fever")) affirmed.push("Fever");

    if (q.includes("headache")) affirmed.push("Headache");
    if (q.includes("nausea")) affirmed.push("Nausea");
    if (q.includes("light") || q.includes("photophobia")) affirmed.push("Photophobia");
    if (q.includes("chest pain") || q.includes("chest tightness")) affirmed.push("Chest Pain");
    if (q.includes("cough")) affirmed.push("Cough");
    if (q.includes("rash")) affirmed.push("Skin Rash");
    if (q.includes("itching")) affirmed.push("Itching");
    if (q.includes("breath") || q.includes("dyspnea")) affirmed.push("Shortness of Breath");

    return { affirmed, denied };
  };

  const { affirmed, denied } = parseTokens();

  return (
    <div className="h-screen max-h-screen overflow-hidden flex flex-col bg-[var(--bg-color)] text-[var(--text-color)]">
      
      {/* COMPACT TOP NAVIGATION BAR */}
      <header className="border-b-2 border-[var(--border-color)] bg-[var(--card-bg)] px-4 py-2.5 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <Link href="/" className="font-['Syne'] text-xl font-extrabold uppercase tracking-tight flex items-center gap-1.5 text-[var(--text-color)] no-underline">
            <span className="text-[var(--accent-blue)]">▲</span> HEALTHTRACK
          </Link>
          <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 border border-[var(--border-color)] bg-white text-black">
            Studio v1.0
          </span>
          <span className="hidden sm:inline text-xs font-mono opacity-60">|</span>
          <span className="hidden sm:inline text-xs font-mono opacity-80">
            Tenant: <b>{user?.email || demoUser?.name || "Patient Alpha"}</b>
          </span>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Active Role/Member Badge */}
          <div className="flex items-center border-2 border-[var(--border-color)] bg-white text-black px-2 py-0.5 gap-1.5 text-[11px] font-bold font-mono">
            <span className="w-2 h-2 rounded-full bg-[var(--accent-green)]"></span>
            <span>Role: {activeMember ? `${activeMember.name} (${activeMember.relationship})` : "Self"}</span>
          </div>

          <ThemeToggle />

          <button
            type="button"
            onClick={logout}
            className="neo-btn !py-1 !px-2.5 !text-[11px] bg-white text-black hover:bg-neutral-100"
            title="Log Out"
          >
            Sign Out
          </button>
        </div>
      </header>

      {/* WORKSPACE LAYOUT: LEFT SIDEBAR + SCROLLABLE MAIN CONTENT (FITS LAPTOP VIEWPORT) */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        
        {/* 2.1 LEFT-SIDE NAVBAR (Fits laptop screen, scrollable) */}
        <aside className="w-full md:w-64 border-b-2 md:border-b-0 md:border-r-2 border-[var(--border-color)] bg-[var(--card-bg)] flex flex-col justify-between shrink-0 h-full overflow-y-auto p-3.5">
          <div>
            {/* Active Role Selector Widget */}
            <div className="p-2.5 border-2 border-[var(--border-color)] bg-white text-black shadow-[2px_2px_0px_var(--shadow-color)] mb-4">
              <label className="font-['Space_Grotesk'] text-[10px] font-bold uppercase opacity-75 block mb-1">
                Active Patient Profile (Role):
              </label>
              <select
                value={activeMember?.member_id || ""}
                onChange={(e) => {
                  const m = members.find((x) => x.member_id === e.target.value);
                  if (m) selectRoleMember(m);
                }}
                className="w-full p-1.5 border border-black bg-neutral-50 text-xs font-bold uppercase outline-none cursor-pointer"
              >
                {members.map((m) => (
                  <option key={m.member_id} value={m.member_id}>
                    {m.name} ({m.relationship} • {m.age || "?"}y)
                  </option>
                ))}
              </select>
            </div>

            {/* Left Nav Menu Items */}
            <nav className="space-y-1.5">
              <button
                type="button"
                onClick={() => setActiveTab("input")}
                className={`w-full text-left p-2.5 border-2 border-[var(--border-color)] font-['Space_Grotesk'] text-xs font-bold uppercase transition-all flex items-center justify-between ${
                  activeTab === "input"
                    ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[3px_3px_0px_var(--shadow-color)] -translate-y-0.5"
                    : "bg-white text-black hover:bg-neutral-100"
                }`}
              >
                <span className="flex items-center gap-2">
                  <ClipboardList className="w-4 h-4" /> 01. Dashboard Input
                </span>
                <span className="neo-badge neo-badge-green text-[9px]">Predict</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("chatbot")}
                className={`w-full text-left p-2.5 border-2 border-[var(--border-color)] font-['Space_Grotesk'] text-xs font-bold uppercase transition-all flex items-center justify-between ${
                  activeTab === "chatbot"
                    ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[3px_3px_0px_var(--shadow-color)] -translate-y-0.5"
                    : "bg-white text-black hover:bg-neutral-100"
                }`}
              >
                <span className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4" /> 02. Telehealth Chatbot
                </span>
                <span className="neo-badge neo-badge-yellow text-[9px]">SSE</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("profile")}
                className={`w-full text-left p-2.5 border-2 border-[var(--border-color)] font-['Space_Grotesk'] text-xs font-bold uppercase transition-all flex items-center justify-between ${
                  activeTab === "profile"
                    ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[3px_3px_0px_var(--shadow-color)] -translate-y-0.5"
                    : "bg-white text-black hover:bg-neutral-100"
                }`}
              >
                <span className="flex items-center gap-2">
                  <User className="w-4 h-4" /> 03. Patient Profile
                </span>
                <span className="text-[10px] font-mono opacity-70">Vault</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("summary")}
                className={`w-full text-left p-2.5 border-2 border-[var(--border-color)] font-['Space_Grotesk'] text-xs font-bold uppercase transition-all flex items-center justify-between ${
                  activeTab === "summary"
                    ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[3px_3px_0px_var(--shadow-color)] -translate-y-0.5"
                    : "bg-white text-black hover:bg-neutral-100"
                }`}
              >
                <span className="flex items-center gap-2">
                  <Activity className="w-4 h-4" /> 04. History Summary
                </span>
                <span className="text-[10px] font-mono opacity-70">Records</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("status")}
                className={`w-full text-left p-2.5 border-2 border-[var(--border-color)] font-['Space_Grotesk'] text-xs font-bold uppercase transition-all flex items-center justify-between ${
                  activeTab === "status"
                    ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[3px_3px_0px_var(--shadow-color)] -translate-y-0.5"
                    : "bg-white text-black hover:bg-neutral-100"
                }`}
              >
                <span className="flex items-center gap-2">
                  <Settings className="w-4 h-4" /> 05. Model Status
                </span>
                <span className="neo-badge text-[9px]">v{healthData?.model_version || 1}</span>
              </button>
            </nav>
          </div>

          {/* Sidebar Footer Info */}
          <div className="pt-3 border-t-2 border-[var(--border-color)] text-[10px] font-mono opacity-75 space-y-1">
            <div>Gateway: <b>{healthData?.status || "HEALTHY"}</b></div>
            <div>Model: <b>{statusData?.active_model || "qwen2.5:0.5b"}</b></div>
            <Link href="/" className="block pt-1 text-[var(--accent-blue)] font-bold underline">
              ← Return to Home Page
            </Link>
          </div>
        </aside>

        {/* MAIN SCROLLABLE DASHBOARD VIEWPORT (Scrollable for general tabs, locked viewport for chatbot) */}
        <main
          className={`flex-1 h-full bg-[var(--card-bg)] ${
            activeTab === "chatbot"
              ? "overflow-hidden p-3 sm:p-4 flex flex-col min-h-0"
              : "overflow-y-auto p-4 sm:p-6"
          }`}
        >
          {dbError && (
            <div className="p-3 mb-4 border-2 border-red-500 bg-red-100 text-red-900 text-xs flex items-center justify-between gap-3 shadow-[2px_2px_0px_rgba(239,68,68,1)]">
              <div className="flex items-center gap-2">
                <span className="font-bold text-red-700">⚠️ Database Notice:</span>
                <span>{dbError}</span>
              </div>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="neo-btn !py-1 !px-2.5 !text-[11px] bg-white text-black hover:bg-neutral-100 shrink-0 font-bold"
              >
                Retry Connection
              </button>
            </div>
          )}
          
          {/* ─────────────────────────────────────────────────────────────
              VIEW 1: INPUT & ROLE-BASED SYMPTOM CHECKER FORM
             ───────────────────────────────────────────────────────────── */}
          {activeTab === "input" && (
            <div>
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-3 border-b-2 border-[var(--border-color)] mb-4 gap-2">
                <div>
                  <h2 className="font-['Space_Grotesk'] text-xl font-extrabold uppercase">
                    Role-Based Symptom Input: {activeMember?.name || "Self"}
                  </h2>
                  <p className="text-xs opacity-75">
                    Pre-populated with {activeMember?.name}&apos;s profile ({activeMember?.age}y, {activeMember?.biological_sex}). Triggers <code>POST /api/predict</code>.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="neo-badge neo-badge-green">Primary Care 99.9%</span>
                  <span className="neo-badge neo-badge-yellow">APS-95</span>
                </div>
              </div>

              {/* Acute-on-Chronic Notification if present in member history */}
              {activeMember?.chronic_conditions && activeMember.chronic_conditions.length > 0 && (
                <div className="p-3 border-2 border-[var(--accent-yellow)] bg-[#fef3c7] text-[#92400e] text-xs mb-4">
                  <span className="font-bold inline-flex items-center gap-1.5 mr-1">
                    <AlertCircle className="w-4 h-4 text-amber-700" />
                    Documented Chronic Baseline:
                  </span>
                  {activeMember.chronic_conditions.join(", ")}. The model actively screens for acute-on-chronic interactions.
                </div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                
                {/* Form Input Area (7 cols) */}
                <div className="lg:col-span-7 space-y-4">
                  {/* Model Architecture Selector */}
                  <div>
                    <div className="font-['Space_Grotesk'] text-[11px] font-bold uppercase mb-1.5 opacity-80 flex items-center justify-between">
                      <span>Target Diagnostic Architecture:</span>
                      <span className="font-mono text-[10px] text-blue-700 font-bold">
                        {modelTier === "auto" ? "Autonomous Clinical Router" : modelTier === "primary_41" ? "41 Primary Conditions (99.9%)" : modelTier === "general_50" ? "50 Multi-System Conditions (93.0%)" : "All 580 Medical Classes"}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                      <button
                        type="button"
                        onClick={() => setModelTier("auto")}
                        className={`p-2 border-2 border-black text-left transition-all ${
                          modelTier === "auto"
                            ? "bg-[var(--accent-color)] text-[var(--accent-text)] font-bold shadow-[2px_2px_0px_#000]"
                            : "bg-white text-black hover:bg-neutral-100"
                        }`}
                      >
                        <div className="text-[10px] font-['Space_Grotesk'] uppercase font-bold">Auto Router</div>
                        <div className="text-[9px] opacity-75 font-mono">Dynamic Ensemble</div>
                      </button>
                      <button
                        type="button"
                        onClick={() => setModelTier("primary_41")}
                        className={`p-2 border-2 border-black text-left transition-all ${
                          modelTier === "primary_41"
                            ? "bg-[var(--accent-color)] text-[var(--accent-text)] font-bold shadow-[2px_2px_0px_#000]"
                            : "bg-white text-black hover:bg-neutral-100"
                        }`}
                      >
                        <div className="text-[10px] font-['Space_Grotesk'] uppercase font-bold">Primary Care</div>
                        <div className="text-[9px] opacity-75 font-mono">41 Classes (99.9%)</div>
                      </button>
                      <button
                        type="button"
                        onClick={() => setModelTier("general_50")}
                        className={`p-2 border-2 border-black text-left transition-all ${
                          modelTier === "general_50"
                            ? "bg-[var(--accent-color)] text-[var(--accent-text)] font-bold shadow-[2px_2px_0px_#000]"
                            : "bg-white text-black hover:bg-neutral-100"
                        }`}
                      >
                        <div className="text-[10px] font-['Space_Grotesk'] uppercase font-bold">General Clinical</div>
                        <div className="text-[9px] opacity-75 font-mono">50 Classes (93.0%)</div>
                      </button>
                      <button
                        type="button"
                        onClick={() => setModelTier("all_580")}
                        className={`p-2 border-2 border-black text-left transition-all ${
                          modelTier === "all_580"
                            ? "bg-[var(--accent-color)] text-[var(--accent-text)] font-bold shadow-[2px_2px_0px_#000]"
                            : "bg-white text-black hover:bg-neutral-100"
                        }`}
                      >
                        <div className="text-[10px] font-['Space_Grotesk'] uppercase font-bold">All Classes</div>
                        <div className="text-[9px] opacity-75 font-mono">580 Universe</div>
                      </button>
                    </div>
                  </div>

                  {/* Active Loaded Condition Banner */}
                  {activeLoadedCondition && (
                    <div className="p-2.5 border-2 border-black bg-[#fef9c3] text-black text-xs font-bold flex items-center justify-between shadow-[2px_2px_0px_#000]">
                      <div className="flex items-center gap-2">
                        <Check className="w-4 h-4 stroke-[3] text-green-700" />
                        <span>
                          Active Directory Condition: <span className="uppercase text-amber-950 font-extrabold">{activeLoadedCondition}</span>
                        </span>
                        <span className="neo-badge neo-badge-green text-[9px]">
                          {modelTier.toUpperCase()}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={handleClearLoadedCondition}
                        className="neo-btn !py-0.5 !px-2 !text-[10px] bg-white text-black hover:bg-neutral-100 flex items-center gap-1"
                      >
                        <X className="w-3 h-3" /> Unload Condition
                      </button>
                    </div>
                  )}

                  {/* Free-Text Narrative */}
                  <div>
                    <label className="font-['Space_Grotesk'] text-xs font-bold uppercase block mb-1">
                      Narrative Symptom Description (Free-Text):
                    </label>
                    <textarea
                      rows={4}
                      value={queryText}
                      onChange={(e) => setQueryText(e.target.value)}
                      className="w-full p-3 border-2 border-[var(--border-color)] bg-white text-black text-xs font-['Inter'] outline-none shadow-[2px_2px_0px_var(--shadow-color)]"
                      placeholder="Describe symptoms in plain words..."
                    />
                  </div>

                  {/* NegEx Token Chips */}
                  <div>
                    <div className="font-['Space_Grotesk'] text-[11px] font-bold uppercase mb-1.5 opacity-80 flex justify-between">
                      <span>Live Clinical Token Parser (NegEx):</span>
                      <span className="font-mono text-[10px]">{affirmed.length} Affirmed | {denied.length} Denied</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {affirmed.map((t) => (
                        <span key={t} className="neo-badge neo-badge-green text-[10px] inline-flex items-center gap-1">
                          <Check className="w-3 h-3" /> {t}
                        </span>
                      ))}
                      {denied.map((t) => (
                        <span key={t} className="neo-badge neo-badge-red line-through text-[10px] inline-flex items-center gap-1">
                          <X className="w-3 h-3" /> {t} (Denied)
                        </span>
                      ))}
                      {affirmed.length === 0 && denied.length === 0 && (
                        <span className="text-[11px] opacity-50 italic">Type symptoms above to preview NegEx extraction...</span>
                      )}
                    </div>
                  </div>

                  {/* Quick Checklist */}
                  <div>
                    <div className="font-['Space_Grotesk'] text-[11px] font-bold uppercase mb-1.5 opacity-80">
                      Symptom Checklist (Click to add to input):
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {CHECKLIST_SYMPTOMS.map((item) => {
                        const isSelected = selectedSymptoms.includes(item.id);
                        return (
                          <button
                            key={item.id}
                            type="button"
                            onClick={() => {
                              setSelectedSymptoms((prev) =>
                                prev.includes(item.id) ? prev.filter((x) => x !== item.id) : [...prev, item.id]
                              );
                            }}
                            className={`neo-btn !py-1 !px-2 !text-[11px] inline-flex items-center gap-1 ${
                              isSelected ? "neo-btn-primary" : "bg-white text-black"
                            }`}
                          >
                            {isSelected ? <Check className="w-3 h-3" /> : <Plus className="w-3 h-3" />}
                            {item.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Properly Arranged Clinical Classes Directory (All 580 Classes) */}
                  <ClinicalClassesBrowser
                    onLoadSymptoms={handleLoadFromDirectory}
                    selectedTier={modelTier}
                    onSelectTier={setModelTier}
                    activeConditionName={activeLoadedCondition}
                    onClearCondition={handleClearLoadedCondition}
                  />

                  {/* Demographic Controls Auto-Synced with Role */}
                  <div className="p-3 border-2 border-[var(--border-color)] bg-white text-black grid grid-cols-3 gap-3 text-xs">
                    <div>
                      <label className="font-bold uppercase font-['Space_Grotesk'] text-[10px] block mb-0.5">Age:</label>
                      <input
                        type="number"
                        min="0"
                        max="125"
                        value={age || ""}
                        onChange={(e) => setAge(e.target.value ? parseFloat(e.target.value) : undefined)}
                        className="w-full p-1.5 border border-black font-mono font-bold"
                      />
                    </div>

                    <div>
                      <label className="font-bold uppercase font-['Space_Grotesk'] text-[10px] block mb-0.5">Biological Sex:</label>
                      <select
                        value={biologicalSex}
                        onChange={(e) => setBiologicalSex(e.target.value)}
                        className="w-full p-1.5 border border-black uppercase font-bold"
                      >
                        <option value="FEMALE">Female</option>
                        <option value="MALE">Male</option>
                        <option value="INTERSEX">Intersex</option>
                      </select>
                    </div>

                    <div className="flex items-center pt-3">
                      <label className="flex items-center gap-1.5 cursor-pointer font-bold uppercase text-[11px]">
                        <input
                          type="checkbox"
                          checked={isPregnant}
                          onChange={(e) => setIsPregnant(e.target.checked)}
                          className="accent-black"
                        />
                        <span>Pregnant</span>
                      </label>
                    </div>
                  </div>

                  {predictError && (
                    <div className="p-2.5 bg-red-100 text-red-900 border border-red-500 text-xs font-semibold">
                      {predictError}
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={handleRunDiagnosis}
                    disabled={isPredicting}
                    className="neo-btn neo-btn-primary w-full justify-center !py-3 text-sm"
                  >
                    {isPredicting ? "Running Clinical Inference..." : "Execute Prediction (POST /api/predict) →"}
                  </button>
                </div>

                {/* Right Column: Diagnostic Output HUD (5 cols) */}
                <div className="lg:col-span-5 space-y-3">
                  <div className="font-['Space_Grotesk'] text-xs font-bold uppercase pb-1 border-b border-[var(--border-color)] flex justify-between">
                    <span>Clinical Diagnostic HUD (Top 3 Differential)</span>
                    {report && <span className="font-mono text-emerald-700">{report.cached ? "Cached < 0.1ms" : `${report.execution_time_ms}ms`}</span>}
                  </div>

                  <EmergencyBanner alert={report?.emergency_alert} />

                  <ActiveInquiryModal
                    inquiry={report?.active_inquiry}
                    onSelectAnswer={(ans) => setQueryText((prev) => `${prev.trim()} (${ans})`)}
                  />

                  {report && (
                    <ConformalSetPill
                      conformalSet={report.conformal_differential_set}
                      coverageGuaranteed={report.conformal_coverage_guaranteed}
                    />
                  )}

                  {report ? (
                    <div className="space-y-3">
                      <div className="p-2.5 border border-black bg-white text-black text-xs">
                        <div className="font-bold uppercase text-[10px] opacity-75">Triage Urgency:</div>
                        <div className="neo-badge neo-badge-green my-1">{report.triage_level}</div>
                        <p className="leading-snug">{report.triage_recommendation}</p>
                      </div>

                      {report.differential_diagnosis.slice(0, 3).map((cand) => (
                        <DiagnosticCard
                          key={cand.condition}
                          explanation={cand}
                          onOpenGuideline={(cond, top, sum) =>
                            setGuidelineModal({ isOpen: true, condition: cond, topic: top, summary: sum })
                          }
                        />
                      ))}
                    </div>
                  ) : (
                    <div className="p-8 border-2 border-dashed border-[var(--border-color)] text-center text-xs opacity-60">
                      Submit patient symptoms on the left to see calibrated conditions, 95% conformal bounds, and ESI triage.
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              VIEW 2: CHATBOT STREAMING CONSULTATION (POST /api/chat/stream)
             ───────────────────────────────────────────────────────────── */}
          {activeTab === "chatbot" && (
            <div className="flex-1 flex flex-col min-h-0 h-full">
              {/* Header with Session selector & Connection Status */}
              <div className="pb-2.5 border-b-2 border-[var(--border-color)] shrink-0 mb-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-3 flex-wrap">
                  <div>
                    <h2 className="font-['Space_Grotesk'] text-base sm:text-lg font-bold uppercase tracking-tight flex items-center gap-2">
                      <MessageSquare className="w-5 h-5 text-[var(--accent-blue)]" />
                      Telehealth Clinical Consultation
                    </h2>
                    <div className="text-[11px] opacity-75 font-mono flex items-center gap-2">
                      <span>Active Patient: <b>{activeMember?.name || "Self"}</b> ({activeMember?.relationship || "Primary"})</span>
                    </div>
                  </div>

                  {/* Session Dropdown Picker */}
                  <div className="flex items-center gap-1.5 bg-white text-black border-2 border-[var(--border-color)] px-2 py-1 shadow-[2px_2px_0px_var(--shadow-color)]">
                    <span className="text-[10px] font-['Space_Grotesk'] font-bold uppercase opacity-60">Session:</span>
                    <select
                      value={activeSession?.session_id || ""}
                      onChange={(e) => handleSelectSession(e.target.value)}
                      className="text-xs font-mono font-bold bg-transparent outline-none cursor-pointer max-w-[150px] truncate"
                    >
                      {sessions.map((s) => (
                        <option key={s.session_id} value={s.session_id}>
                          {s.title || s.session_id.slice(0, 14)}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    type="button"
                    onClick={handleCreateNewSession}
                    className="neo-btn !py-1 !px-2.5 !text-[11px] bg-white text-black inline-flex items-center gap-1 hover:bg-neutral-100"
                    title="Start a fresh consultation session"
                  >
                    <Plus className="w-3.5 h-3.5" /> New Session
                  </button>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span
                    className={`neo-badge inline-flex items-center gap-1.5 ${
                      isChatStreaming ? "neo-badge-green animate-pulse" : "bg-neutral-200 text-neutral-800"
                    }`}
                  >
                    <span className={`w-2 h-2 rounded-full ${isChatStreaming ? "bg-emerald-600 animate-ping" : "bg-neutral-500"}`} />
                    {isChatStreaming ? "Streaming Tokens..." : "SSE Channel Ready"}
                  </span>
                </div>
              </div>

              {/* Streaming Clinical Diagnostic HUD */}
              <ChatLiveDiagnosticHUD meta={chatMetadata} />

              {/* Chat Feed (Scrolls smoothly within bounds) */}
              <div
                ref={chatFeedRef}
                className="flex-1 overflow-y-auto min-h-0 pr-1 sm:pr-2 space-y-3"
              >
                {chatMessages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center p-6 text-center border-2 border-dashed border-[var(--border-color)] bg-neutral-50/50">
                    <MessageSquare className="w-8 h-8 opacity-30 mb-2" />
                    <p className="font-['Space_Grotesk'] font-bold text-sm uppercase mb-1">
                      Start Consultation for {activeMember?.name || "Self"}
                    </p>
                    <p className="text-xs opacity-75 max-w-md mb-4">
                      Describe onset, severity, and associated symptoms. The AI pipeline runs NegEx token extraction, clinical classifier inference, and RAG retrieval across NIH MedlinePlus.
                    </p>

                    <div className="w-full max-w-lg space-y-2 text-left">
                      <div className="text-[10px] font-mono font-bold uppercase opacity-60">
                        Quick Clinical Presentation Starters:
                      </div>
                      {[
                        "Severe throbbing unilateral headache with nausea and sensitivity to light.",
                        "Sudden onset sharp chest tightness when breathing deeply, with dry cough.",
                        "High fever (102°F) with body chills, fatigue, and muscle aches for 2 days.",
                      ].map((prompt, idx) => (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => {
                            setChatInput(prompt);
                          }}
                          className="w-full text-left p-2 border border-black bg-white hover:bg-neutral-100 text-xs font-mono transition-all flex items-center justify-between"
                        >
                          <span className="truncate">{prompt}</span>
                          <span className="text-[10px] text-[var(--accent-blue)] font-bold shrink-0 ml-2">Load →</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  chatMessages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.sender === "patient" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[88%] sm:max-w-[82%] border-2 border-[var(--border-color)] p-3 shadow-[3px_3px_0px_var(--shadow-color)] text-xs leading-relaxed ${
                          msg.sender === "patient"
                            ? "bg-[var(--accent-color)] text-[var(--accent-text)]"
                            : "bg-white text-black"
                        }`}
                      >
                        <div className="text-[10px] font-mono uppercase opacity-75 pb-1 mb-1.5 border-b border-current flex items-center justify-between gap-2">
                          <span className="font-bold flex items-center gap-1">
                            {msg.sender === "patient" ? (
                              <>
                                <User className="w-3 h-3" />
                                {activeMember?.name || "Patient"} ({activeMember?.relationship || "Self"})
                              </>
                            ) : (
                              <>
                                <Activity className="w-3 h-3 text-[var(--accent-blue)]" />
                                HealthTrack Clinical Assistant (Qwen-Med SFT)
                              </>
                            )}
                          </span>
                          {msg.attachedDoc && (
                            <span className="inline-flex items-center gap-1 bg-black/10 px-1.5 py-0.5 rounded text-[9px] font-mono">
                              <Paperclip className="w-3 h-3" /> {msg.attachedDoc}
                            </span>
                          )}
                        </div>

                        {msg.sender === "assistant" ? (
                          <FormattedChatMessage
                            text={msg.text}
                            sender={msg.sender}
                            isStreaming={msg.isStreaming}
                          />
                        ) : (
                          <div className="whitespace-pre-wrap">{msg.text}</div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                <div ref={chatBottomRef} />
              </div>

              {/* Sticky Input Bar at Bottom */}
              <div className="pt-2.5 border-t-2 border-[var(--border-color)] shrink-0 mt-2">
                {attachedFile && (
                  <div className="text-xs font-mono mb-2 p-1.5 bg-neutral-100 border border-black flex items-center justify-between">
                    <span className="flex items-center gap-1.5 truncate">
                      <Paperclip className="w-3.5 h-3.5 text-blue-700" />
                      Attached File: <b>{attachedFile.name}</b>
                    </span>
                    <button
                      type="button"
                      onClick={() => setAttachedFile(null)}
                      className="text-red-700 font-bold inline-flex items-center gap-1 hover:opacity-80 shrink-0 ml-2"
                    >
                      <X className="w-3.5 h-3.5" /> Remove
                    </button>
                  </div>
                )}

                <div className="flex gap-2">
                  <label
                    className="neo-btn !px-3 cursor-pointer bg-white text-black inline-flex items-center justify-center shrink-0"
                    title="Attach Lab PDF, EKG, or Rash Photo"
                  >
                    <Paperclip className="w-4 h-4" />
                    <input
                      type="file"
                      accept="application/pdf,image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </label>

                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSendChatStream()}
                    placeholder={`Describe clinical symptoms for ${activeMember?.name || "patient"} or reply to high-yield questions...`}
                    className="flex-1 p-2.5 border-2 border-[var(--border-color)] bg-white text-black text-xs font-['Inter'] outline-none"
                    disabled={isChatStreaming}
                  />

                  <button
                    type="button"
                    onClick={() => handleSendChatStream()}
                    disabled={isChatStreaming || !chatInput.trim()}
                    className="neo-btn neo-btn-primary shrink-0"
                  >
                    {isChatStreaming ? "Streaming..." : "Send Stream →"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              VIEW 3: PATIENT PROFILE & FAMILY MEMBERS VAULT
             ───────────────────────────────────────────────────────────── */}
          {activeTab === "profile" && (
            <div className="space-y-6">
              <div className="pb-3 border-b-2 border-[var(--border-color)] flex items-center justify-between">
                <div>
                  <h2 className="font-['Space_Grotesk'] text-xl font-bold uppercase">
                    Patient Profile &amp; Family Member Management
                  </h2>
                  <p className="text-xs opacity-75">
                    Isolated under <code>users/&#123;uid&#125;/members/&#123;member_id&#125;</code>.
                  </p>
                </div>
              </div>

              {/* Member Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {members.map((m) => {
                  const isCurrent = activeMember?.member_id === m.member_id;
                  return (
                    <div
                      key={m.member_id}
                      onClick={() => selectRoleMember(m)}
                      className={`p-4 border-2 border-[var(--border-color)] cursor-pointer transition-all ${
                        isCurrent
                          ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[4px_4px_0px_var(--shadow-color)]"
                          : "bg-white text-black hover:bg-neutral-50"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="neo-badge neo-badge-yellow text-[10px]">{m.relationship}</span>
                        <span className="text-[10px] font-mono">{m.member_id}</span>
                      </div>
                      <h4 className="font-['Space_Grotesk'] text-lg font-bold uppercase">{m.name}</h4>
                      <p className="text-xs opacity-75 mt-0.5">{m.age || "?"}y • {m.biological_sex}</p>
                      <div className="mt-2 text-xs font-medium border-t border-current pt-1.5 line-clamp-1">
                        Chronic: {m.chronic_conditions.length > 0 ? m.chronic_conditions.join(", ") : "None"}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Register New Member Form */}
              <form onSubmit={handleCreateMember} className="p-4 border-2 border-[var(--border-color)] bg-white text-black shadow-[3px_3px_0px_var(--shadow-color)]">
                <h3 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-3">
                  + Register New Family Member Profile
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs mb-3">
                  <div>
                    <label className="font-bold uppercase block mb-1">Name / Label:</label>
                    <input
                      type="text"
                      required
                      placeholder="e.g. Mom"
                      value={newMemberName}
                      onChange={(e) => setNewMemberName(e.target.value)}
                      className="w-full p-2 border border-black font-['Inter']"
                    />
                  </div>
                  <div>
                    <label className="font-bold uppercase block mb-1">Relationship:</label>
                    <select
                      value={newMemberRel}
                      onChange={(e) => setNewMemberRel(e.target.value)}
                      className="w-full p-2 border border-black font-bold uppercase"
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
                      value={newMemberAge}
                      onChange={(e) => setNewMemberAge(parseInt(e.target.value) || 0)}
                      className="w-full p-2 border border-black font-mono"
                    />
                  </div>
                  <div>
                    <label className="font-bold uppercase block mb-1">Biological Sex:</label>
                    <select
                      value={newMemberSex}
                      onChange={(e) => setNewMemberSex(e.target.value)}
                      className="w-full p-2 border border-black font-bold uppercase"
                    >
                      <option value="MALE">Male</option>
                      <option value="FEMALE">Female</option>
                      <option value="INTERSEX">Intersex</option>
                    </select>
                  </div>
                </div>

                <div className="text-xs mb-3">
                  <label className="font-bold uppercase block mb-1">Known Chronic Diagnoses (Comma separated):</label>
                  <input
                    type="text"
                    placeholder="e.g. Asthma, Hypertension"
                    value={newMemberChronic}
                    onChange={(e) => setNewMemberChronic(e.target.value)}
                    className="w-full p-2 border border-black"
                  />
                </div>

                <button type="submit" className="neo-btn neo-btn-primary !py-2 !px-4 text-xs">
                  Save New Profile
                </button>
              </form>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              VIEW 4: SUMMARY & LONGITUDINAL HEALTH TIMELINE
             ───────────────────────────────────────────────────────────── */}
          {activeTab === "summary" && (
            <div className="space-y-6">
              <div className="pb-3 border-b-2 border-[var(--border-color)] flex items-center justify-between">
                <div>
                  <h2 className="font-['Space_Grotesk'] text-xl font-bold uppercase">
                    Longitudinal History &amp; Consultation Summary
                  </h2>
                  <p className="text-xs opacity-75">
                    Chronological milestone ledger for <b>{activeMember?.name || "Self"}</b>.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => window.print()}
                  className="neo-btn bg-white text-black !py-1 !px-3 !text-xs inline-flex items-center gap-1.5"
                >
                  <Printer className="w-3.5 h-3.5" /> Export PDF Summary
                </button>
              </div>

              {/* Past Diagnoses Timeline */}
              <div className="border-2 border-[var(--border-color)] p-4 bg-white text-black shadow-[3px_3px_0px_var(--shadow-color)]">
                <h3 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-3">
                  Past Encounters &amp; Clinical Milestones:
                </h3>

                {(!activeMember?.longitudinal_summary || activeMember.longitudinal_summary.length === 0) ? (
                  <div className="p-4 border border-dashed border-black/40 text-xs text-center opacity-60">
                    No past clinical milestones logged yet for this member.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {activeMember.longitudinal_summary.map((entry, idx) => (
                      <div key={idx} className="p-2.5 border border-black bg-neutral-50 flex items-center justify-between text-xs font-mono">
                        <div>
                          <span className="font-bold text-black font-['Space_Grotesk'] uppercase mr-2">{entry.condition}</span>
                          <span className="opacity-60">{entry.diagnosed_date}</span>
                        </div>
                        <span className="neo-badge neo-badge-yellow text-[10px]">{entry.status}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Add Milestone Form */}
                <form onSubmit={handleAddCondition} className="mt-4 pt-4 border-t border-black/20 flex gap-2">
                  <input
                    type="text"
                    required
                    placeholder="Diagnosed condition (e.g. Coronary Artery Disease)..."
                    value={newConditionName}
                    onChange={(e) => setNewConditionName(e.target.value)}
                    className="flex-1 p-2 border border-black text-xs"
                  />
                  <input
                    type="date"
                    value={newConditionDate}
                    onChange={(e) => setNewConditionDate(e.target.value)}
                    className="p-2 border border-black text-xs font-mono"
                  />
                  <button type="submit" className="neo-btn !py-2 !px-3 !text-xs bg-black text-white">
                    + Add Milestone
                  </button>
                </form>
              </div>

              {/* Consultation Session Archives */}
              <div className="border-2 border-[var(--border-color)] p-4 bg-white text-black shadow-[3px_3px_0px_var(--shadow-color)]">
                <h3 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-3">
                  Consultation Session Archives ({sessions.length}):
                </h3>
                <div className="space-y-2 max-h-[240px] overflow-y-auto">
                  {sessions.map((s) => (
                    <div key={s.session_id} className="p-2.5 border border-black bg-neutral-50 flex items-center justify-between text-xs">
                      <div>
                        <span className="font-bold font-['Space_Grotesk'] uppercase mr-2">{s.title}</span>
                        <span className="font-mono text-[11px] opacity-70">({s.member_name})</span>
                      </div>
                      <span className="font-mono text-[10px] opacity-75">{s.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              VIEW 5: SYSTEM TELEMETRY & CONTINUAL OPS
             ───────────────────────────────────────────────────────────── */}
          {activeTab === "status" && (
            <div className="space-y-6">
              <div className="pb-3 border-b-2 border-[var(--border-color)] flex items-center justify-between">
                <div>
                  <h2 className="font-['Space_Grotesk'] text-xl font-bold uppercase">
                    Model Serving Status &amp; Continual Learning Ops
                  </h2>
                  <p className="text-xs opacity-75">
                    Live telemetry from <code>GET /health</code> and <code>GET /status</code>.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button type="button" onClick={handleTriggerLearning} className="neo-btn neo-btn-primary !py-1.5 !px-3 !text-xs">
                    Force Learning Cycle
                  </button>
                  <button type="button" onClick={handleRollback} className="neo-btn neo-btn-danger !py-1.5 !px-3 !text-xs">
                    Rollback to Anchor
                  </button>
                </div>
              </div>

              {opsMessage && (
                <div className="p-3 bg-neutral-100 border-2 border-black text-xs font-mono font-bold">
                  {opsMessage}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border-2 border-[var(--border-color)] p-4 bg-white text-black shadow-[3px_3px_0px_var(--shadow-color)]">
                  <h3 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-3">Serving Manager Telemetry</h3>
                  <div className="space-y-2 text-xs font-mono">
                    <div className="flex justify-between border-b pb-1"><span>Service:</span><b>{healthData?.service || "continual-rag-llm"}</b></div>
                    <div className="flex justify-between border-b pb-1"><span>Status:</span><b className="text-emerald-700">{healthData?.status || "HEALTHY"}</b></div>
                    <div className="flex justify-between border-b pb-1"><span>Model Version:</span><b>v{healthData?.model_version || 1}</b></div>
                    <div className="flex justify-between border-b pb-1"><span>Active Model:</span><b>{statusData?.active_model || "qwen2.5:0.5b"}</b></div>
                    <div className="flex justify-between"><span>Active Readers:</span><b>{statusData?.serving_manager?.active_readers ?? 0}</b></div>
                  </div>
                </div>

                <div className="border-2 border-[var(--border-color)] p-4 bg-white text-black shadow-[3px_3px_0px_var(--shadow-color)]">
                  <h3 className="font-['Space_Grotesk'] text-sm font-bold uppercase mb-3">Experience Replay Buffer</h3>
                  <p className="text-xs opacity-75 mb-3 font-['Inter']">Maintains strict 70/30 mixture ratio guarding against catastrophic forgetting.</p>
                  <div className="grid grid-cols-2 gap-2 text-center font-mono">
                    <div className="p-2 border border-black bg-emerald-50">
                      <div className="text-[10px] font-bold uppercase">Anchor Ratio</div>
                      <div className="text-xl font-bold">70.0%</div>
                    </div>
                    <div className="p-2 border border-black bg-blue-50">
                      <div className="text-[10px] font-bold uppercase">Web Ratio</div>
                      <div className="text-xl font-bold">30.0%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>

      {/* NIH MedQuAD Evidence Drawer */}
      <MedQuADGuidelineDrawer
        isOpen={guidelineModal.isOpen}
        condition={guidelineModal.condition}
        topic={guidelineModal.topic}
        summary={guidelineModal.summary}
        onClose={() => setGuidelineModal({ isOpen: false, condition: "" })}
      />

    </div>
  );
}

