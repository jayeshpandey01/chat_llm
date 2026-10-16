"use client";

import { useEffect, useState, useRef } from "react";
import { createSession, listSessions, listMembers, uploadMedicalFile } from "@/lib/api";
import { streamChatConsultation } from "@/lib/sse";
import { SessionRecord, SSEMetadataEvent, MemberProfile } from "@/lib/types";
import ChatLiveDiagnosticHUD from "@/components/checker/ChatLiveDiagnosticHUD";
import FormattedChatMessage from "@/components/checker/FormattedChatMessage";
import { 
  Paperclip, 
  X, 
  AlertTriangle, 
  Send, 
  BrainCircuit, 
  User, 
  Users, 
  Plus, 
  Activity, 
  FileText, 
  ChevronDown, 
  ChevronUp,
  ExternalLink,
  ShieldCheck,
  CheckCircle2
} from "lucide-react";

interface ChatMessage {
  id: string;
  sender: "patient" | "assistant";
  text: string;
  thinking?: string;
  meta?: SSEMetadataEvent;
  attachedDoc?: string;
  isStreaming?: boolean;
}

const COMMON_SYMPTOMS = [
  "fever",
  "cough",
  "chest pain",
  "shortness of breath",
  "headache",
  "chills",
  "sweats",
  "fatigue",
  "skin rash",
  "nausea",
];

export default function ConsultationPage() {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [activeSession, setActiveSession] = useState<SessionRecord | null>(null);
  const [members, setMembers] = useState<MemberProfile[]>([]);
  const [selectedMemberId, setSelectedMemberId] = useState<string>("mem_primary");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputQuery, setInputQuery] = useState("");
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeMetadata, setActiveMetadata] = useState<SSEMetadataEvent | null>(null);
  const [attachedFile, setAttachedFile] = useState<{ id: string; name: string; summary?: string } | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showThinkingTrace, setShowThinkingTrace] = useState<Record<string, boolean>>({});

  const chatBottomRef = useRef<HTMLDivElement>(null);
  const chatFeedRef = useRef<HTMLDivElement>(null);

  // Load family members and sessions on mount
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [membersRes, sessionsRes] = await Promise.all([
          listMembers().catch(() => ({ status: "MOCK", members: [], count: 0 })),
          listSessions().catch(() => ({ status: "MOCK", sessions: [], count: 0 })),
        ]);

        const loadedMembers = membersRes.members || [];
        setMembers(loadedMembers);

        const loadedSessions = sessionsRes.sessions || [];
        setSessions(loadedSessions);

        if (loadedSessions.length > 0) {
          selectSession(loadedSessions[0]);
        } else {
          // Auto create a primary session if none exist
          const newSess = await createSession({
            title: "Primary Clinical Encounter",
            member_id: "mem_primary",
          });
          setSessions([newSess.session]);
          selectSession(newSess.session);
        }
      } catch (err) {
        console.warn("Initial load warning:", err);
      }
    }
    loadInitialData();
  }, []);

  const selectSession = (sess: SessionRecord) => {
    setActiveSession(sess);
    setSelectedMemberId(sess.member_id || "mem_primary");
    const mapped: ChatMessage[] = (sess.recent_context_window || []).flatMap((turn, i) => [
      { id: `u-${i}`, sender: "patient", text: turn.user },
      { id: `a-${i}`, sender: "assistant", text: turn.assistant },
    ]);
    setMessages(mapped);

    if (sess.active_differential) {
      setActiveMetadata({
        type: "diagnostic_report",
        member_id: sess.member_id,
        member_name: sess.member_name,
        member_relationship: sess.member_relationship,
        top_condition: sess.active_differential.top_condition,
        top_confidence: sess.active_differential.top_confidence,
        triage_level: sess.active_differential.triage_level,
        triage_recommendation: sess.active_differential.triage_recommendation,
        conformal_set: sess.active_differential.conformal_differential_set,
        is_emergency: !!sess.active_differential.emergency_alert?.is_emergency,
        is_acute_on_chronic: false,
      });
    } else {
      setActiveMetadata(null);
    }
  };

  const handleCreateNewSession = async () => {
    try {
      const activeMember = members.find((m) => m.member_id === selectedMemberId);
      const memberName = activeMember ? activeMember.name : "Self";
      const title = `${memberName} Consultation - ${new Date().toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      })}`;

      const res = await createSession({
        title,
        member_id: selectedMemberId,
        initial_symptoms: selectedSymptoms,
      });

      setSessions((prev) => [res.session, ...prev]);
      selectSession(res.session);
    } catch (err: any) {
      alert(`Error creating session: ${err.message}`);
    }
  };

  const toggleSymptom = (sym: string) => {
    setSelectedSymptoms((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      const isPdf = file.type.includes("pdf") || file.name.endsWith(".pdf");
      const docType = isPdf ? "LAB_REPORT" : "DERMATOLOGY_IMAGE";
      const res = await uploadMedicalFile(file, docType);

      setAttachedFile({
        id: res.file.file_id,
        name: res.file.filename,
        summary: res.file.extracted_clinical_summary,
      });
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  };

  const handleSendStream = async () => {
    if ((!inputQuery.trim() && selectedSymptoms.length === 0) || !activeSession || isStreaming) {
      return;
    }

    const query = inputQuery.trim() || `Patient presenting with: ${selectedSymptoms.join(", ")}`;
    const symptomsToSend = [...selectedSymptoms];
    
    setInputQuery("");
    setSelectedSymptoms([]);
    setIsStreaming(true);

    const userMsgId = `u-${Date.now()}`;
    const assistantMsgId = `a-${Date.now()}`;

    // Add user message & placeholder streaming assistant message
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        sender: "patient",
        text: query,
        attachedDoc: attachedFile?.name,
      },
      {
        id: assistantMsgId,
        sender: "assistant",
        text: "",
        thinking: "",
        isStreaming: true,
      },
    ]);

    // Open thinking drawer by default for the active stream
    setShowThinkingTrace((prev) => ({ ...prev, [assistantMsgId]: true }));

    await streamChatConsultation({
      sessionId: activeSession.session_id,
      queryText: query,
      selectedSymptoms: symptomsToSend,
      documentId: attachedFile?.id,
      onDiagnosticReport: (meta) => {
        setActiveMetadata(meta);
      },
      onThinking: (thinkingChunk) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, thinking: (m.thinking || "") + thinkingChunk }
              : m
          )
        );
        if (chatFeedRef.current) {
          chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
        }
      },
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, text: m.text + token } : m
          )
        );
        if (chatFeedRef.current) {
          chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
        }
      },
      onDone: () => {
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, isStreaming: false } : m
          )
        );
        setAttachedFile(null);
      },
      onError: (err) => {
        console.error("Stream error:", err);
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  text: `${m.text}\n\n*[Clinical Gateway notice: Stream encountered interruption: ${err.message}]*`,
                  isStreaming: false,
                }
              : m
          )
        );
      },
    });
  };

  const toggleThinkingForMessage = (msgId: string) => {
    setShowThinkingTrace((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 h-[calc(100vh-65px)] overflow-hidden bg-[var(--background)]">
      
      {/* ── Left Sidebar: Patient Selector & Session History (3 cols) ───────── */}
      <div className="lg:col-span-3 border-b-2 lg:border-b-0 lg:border-r-2 border-[var(--border-color)] p-4 flex flex-col justify-between bg-[var(--card-bg)] h-full overflow-y-auto">
        <div>
          {/* Family Member Role Switcher */}
          <div className="pb-3 border-b-2 border-[var(--border-color)] mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-[10px] font-['Space_Grotesk'] font-bold uppercase tracking-wider text-neutral-600 flex items-center gap-1.5">
                <Users className="w-3.5 h-3.5 text-blue-600" />
                Active Family Member
              </label>
              <span className="neo-badge text-[9px]">Isolated Context</span>
            </div>

            <select
              value={selectedMemberId}
              onChange={(e) => setSelectedMemberId(e.target.value)}
              className="w-full p-2 border-2 border-[var(--border-color)] bg-white text-black font-['Space_Grotesk'] text-xs font-bold uppercase outline-none shadow-[2px_2px_0px_#000]"
            >
              <option value="mem_primary">Primary Patient (Self)</option>
              {members
                .filter((m) => m.member_id !== "mem_primary")
                .map((m) => (
                  <option key={m.member_id} value={m.member_id}>
                    {m.name} ({m.relationship})
                  </option>
                ))}
            </select>
          </div>

          {/* Consultation Session Controls */}
          <div className="flex items-center justify-between pb-2 mb-3">
            <h3 className="font-['Space_Grotesk'] text-xs font-bold uppercase tracking-tight flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-emerald-600" />
              Past Consultations
            </h3>
            <button
              type="button"
              onClick={handleCreateNewSession}
              className="neo-btn !py-1 !px-2.5 !text-[11px] inline-flex items-center gap-1"
            >
              <Plus className="w-3 h-3" /> New
            </button>
          </div>

          <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1">
            {sessions.length === 0 ? (
              <div className="text-center p-4 text-[11px] opacity-60 italic">
                No past encounters found. Click &quot;+ New&quot; to begin.
              </div>
            ) : (
              sessions.map((sess) => {
                const isActive = activeSession?.session_id === sess.session_id;
                return (
                  <div
                    key={sess.session_id}
                    onClick={() => selectSession(sess)}
                    className={`p-3 border-2 border-[var(--border-color)] cursor-pointer transition-all ${
                      isActive
                        ? "bg-[var(--accent-color)] text-[var(--accent-text)] shadow-[3px_3px_0px_var(--shadow-color)] font-bold"
                        : "bg-white text-black hover:bg-neutral-50 shadow-[1px_1px_0px_#000]"
                    }`}
                  >
                    <div className="text-[10px] font-mono uppercase opacity-85 flex justify-between">
                      <span className="truncate max-w-[140px] font-semibold">
                        {sess.member_name} ({sess.member_relationship})
                      </span>
                      <span>{sess.status || "ACTIVE"}</span>
                    </div>
                    <div className="font-['Space_Grotesk'] text-xs font-bold uppercase mt-1 truncate">
                      {sess.title || "Consultation Encounter"}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer Guarantee */}
        <div className="pt-3 border-t-2 border-[var(--border-color)] text-[10px] font-mono opacity-80 flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
          <span>Zero-waste 1-Read / 1-Write persistence</span>
        </div>
      </div>

      {/* ── Center Column: Real-Time Consultation Stream (6 cols) ──────────── */}
      <div className="lg:col-span-6 border-b-2 lg:border-b-0 lg:border-r-2 border-[var(--border-color)] p-3 sm:p-4 flex flex-col min-h-0 h-full bg-white">
        {/* Active Session Header Banner */}
        <div className="shrink-0 flex items-center justify-between pb-2.5 border-b-2 border-[var(--border-color)] mb-2.5">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-['Space_Grotesk'] text-base font-bold uppercase">
                {activeSession?.title || "Active Telehealth Consultation"}
              </h2>
            </div>
            <div className="text-xs opacity-75 font-mono">
              Patient: <b>{activeSession?.member_name || "Self"}</b> • Member ID: {activeSession?.member_id || "mem_primary"}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className={`neo-badge ${isStreaming ? "neo-badge-green animate-pulse" : ""}`}>
              {isStreaming ? "Streaming SSE" : "Encounter Ready"}
            </span>
          </div>
        </div>

        {/* Emergency / Acute-on-Chronic Top Alert Banner (if fired) */}
        {activeMetadata?.is_emergency && (
          <div className="shrink-0 p-2.5 border-2 border-red-600 bg-red-100 text-red-950 font-bold mb-2 shadow-[2px_2px_0px_#991b1b] flex items-center gap-2 text-xs">
            <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
            <div>
              <span className="font-['Space_Grotesk'] uppercase tracking-wider block font-black text-[11px]">
                CRITICAL EMERGENCY LEVEL TRIGGERED (ESI 1/2)
              </span>
              <span>Immediate emergency intervention required. Do not wait for a clinic appointment.</span>
            </div>
          </div>
        )}

        {activeMetadata?.is_acute_on_chronic && activeMetadata.acute_on_chronic_alert && (
          <div className="shrink-0 p-2.5 border-2 border-amber-600 bg-amber-50 text-amber-950 font-bold mb-2 shadow-[2px_2px_0px_#b45309] flex items-center gap-2 text-xs">
            <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
            <div>
              <span className="font-['Space_Grotesk'] uppercase tracking-wider block font-black text-amber-800 text-[11px]">
                LONGITUDINAL ACUTE-ON-CHRONIC ALERT
              </span>
              <span>{activeMetadata.acute_on_chronic_alert}</span>
            </div>
          </div>
        )}

        {/* Chat Message Scrollable Feed (Fills remaining height, no cutting off!) */}
        <div
          ref={chatFeedRef}
          className="flex-1 overflow-y-auto min-h-0 pr-1 sm:pr-2 space-y-3"
        >
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center p-8 text-center border-2 border-dashed border-neutral-300 text-xs opacity-60">
              <Activity className="w-8 h-8 mx-auto mb-2 text-neutral-400" />
              <p className="font-['Space_Grotesk'] font-bold uppercase">No clinical dialogue recorded yet.</p>
              <p className="font-mono text-[11px] mt-1">Select symptoms below or type a symptom presentation to begin real-time streaming.</p>
            </div>
          ) : (
            messages.map((msg) => {
              const isUser = msg.sender === "patient";
              const isThinkingOpen = showThinkingTrace[msg.id];

              return (
                <div key={msg.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[88%] sm:max-w-[82%] border-2 border-black p-3 shadow-[3px_3px_0px_#000] text-xs leading-relaxed ${
                      isUser
                        ? "bg-[var(--accent-color)] text-[var(--accent-text)]"
                        : "bg-white text-black"
                    }`}
                  >
                    {/* Message Header */}
                    <div className="flex items-center justify-between text-[10px] font-mono uppercase opacity-75 pb-1 mb-1.5 border-b border-current">
                      <span className="font-bold flex items-center gap-1">
                        {isUser ? (
                          <>
                            <User className="w-3 h-3" />
                            Patient ({activeSession?.member_name || "Self"})
                          </>
                        ) : (
                          <>
                            <Activity className="w-3 h-3 text-[var(--accent-blue)]" />
                            HealthTrack Clinical Decision Support
                          </>
                        )}
                      </span>
                      {msg.attachedDoc && (
                        <span className="inline-flex items-center gap-1 font-mono text-[9px] bg-black/10 px-1.5 py-0.5 rounded">
                          <Paperclip className="w-3 h-3" /> {msg.attachedDoc}
                        </span>
                      )}
                    </div>

                    {/* Reasoning / Thinking Trace Accordion (When reasoning model is active) */}
                    {!isUser && msg.thinking && (
                      <div className="mb-2 border border-amber-400 bg-amber-50/80 rounded p-2 text-xs">
                        <button
                          type="button"
                          onClick={() => toggleThinkingForMessage(msg.id)}
                          className="w-full flex items-center justify-between text-[10px] font-['Space_Grotesk'] font-bold uppercase text-amber-900 hover:opacity-80"
                        >
                          <span className="flex items-center gap-1.5">
                            <BrainCircuit className="w-3.5 h-3.5 text-amber-700 animate-pulse" />
                            Clinical Reasoning Trace {msg.isStreaming && !msg.text ? "(Reasoning in progress...)" : ""}
                          </span>
                          {isThinkingOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                        </button>
                        
                        {isThinkingOpen && (
                          <div className="mt-1.5 pt-1.5 border-t border-amber-200 text-[10px] font-mono text-neutral-800 whitespace-pre-wrap max-h-36 overflow-y-auto leading-relaxed">
                            {msg.thinking}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Main Clinical Output */}
                    {isUser ? (
                      <div className="text-xs leading-relaxed whitespace-pre-wrap font-['Inter']">
                        {msg.text}
                      </div>
                    ) : (
                      <FormattedChatMessage
                        text={msg.text}
                        sender="assistant"
                        isStreaming={msg.isStreaming}
                      />
                    )}
                  </div>
                </div>
              );
            })
          )}
          <div ref={chatBottomRef} />
        </div>

        {/* ── Input Bar, Quick Symptoms & Document Uploader ─────────────────── */}
        <div className="shrink-0 pt-2.5 border-t-2 border-[var(--border-color)] mt-2">
          {/* Quick Symptoms Multi-Select Tags */}
          <div className="mb-2">
            <div className="text-[10px] font-['Space_Grotesk'] font-bold uppercase text-neutral-500 mb-1 flex items-center justify-between">
              <span>Quick-Pick Common Symptoms:</span>
              {selectedSymptoms.length > 0 && (
                <button
                  type="button"
                  onClick={() => setSelectedSymptoms([])}
                  className="text-red-700 font-mono text-[9px] hover:underline"
                >
                  Clear Selection ({selectedSymptoms.length})
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-1">
              {COMMON_SYMPTOMS.map((sym) => {
                const isSelected = selectedSymptoms.includes(sym);
                return (
                  <button
                    key={sym}
                    type="button"
                    onClick={() => toggleSymptom(sym)}
                    className={`text-[10px] font-['Space_Grotesk'] font-bold uppercase px-2 py-0.5 border border-black transition-all ${
                      isSelected
                        ? "bg-black text-white shadow-[1px_1px_0px_#000]"
                        : "bg-neutral-100 text-black hover:bg-neutral-200"
                    }`}
                  >
                    {isSelected ? `✓ ${sym}` : `+ ${sym}`}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Attached Document Preview Pill */}
          {attachedFile && (
            <div className="text-xs font-mono mb-2 p-2 bg-emerald-50 border border-emerald-600 text-emerald-950 flex items-center justify-between shadow-[2px_2px_0px_#065f46]">
              <div className="flex items-center gap-2 truncate">
                <FileText className="w-4 h-4 text-emerald-700 shrink-0" />
                <span className="truncate">
                  Attached: <b>{attachedFile.name}</b> {attachedFile.summary ? `• (${attachedFile.summary.slice(0, 40)}...)` : ""}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setAttachedFile(null)}
                className="text-red-700 font-bold hover:opacity-80 p-0.5"
                title="Remove attachment"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Input Controls */}
          <div className="flex gap-2">
            <label
              className={`neo-btn !px-3 cursor-pointer inline-flex items-center justify-center ${
                isUploading ? "opacity-50 cursor-wait" : ""
              }`}
              title="Upload Lab PDF (CBC/CMP/Troponin) or Clinical Skin Photo"
            >
              <Paperclip className="w-4 h-4" />
              <input
                type="file"
                accept="application/pdf,image/*"
                onChange={handleFileUpload}
                disabled={isUploading || isStreaming}
                className="hidden"
              />
            </label>

            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSendStream()}
              disabled={isStreaming}
              placeholder="Describe symptoms, duration, or acute changes..."
              className="flex-1 p-2.5 border-2 border-[var(--border-color)] bg-white text-black text-xs font-['Inter'] outline-none shadow-[2px_2px_0px_#000] focus:border-black"
            />

            <button
              type="button"
              onClick={handleSendStream}
              disabled={isStreaming || (!inputQuery.trim() && selectedSymptoms.length === 0)}
              className="neo-btn neo-btn-primary inline-flex items-center gap-1.5"
            >
              {isStreaming ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-white animate-ping"></span>
                  Streaming...
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  Send Stream
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* ── Right Column: Live Clinical Heads-Up Display (HUD) (3 cols) ────── */}
      <div className="lg:col-span-3 p-4 sm:p-5 bg-[var(--card-bg)] flex flex-col justify-between h-full overflow-y-auto">
        <div className="space-y-3">
          <ChatLiveDiagnosticHUD meta={activeMetadata} />

          {/* Retrieved Evidence Citations (NIH MedlinePlus & PubMed) */}
          {activeMetadata?.retrieved_evidence && activeMetadata.retrieved_evidence.length > 0 && (
            <div className="p-3 border-2 border-black bg-white shadow-[3px_3px_0px_#000]">
              <div className="text-[10px] font-['Space_Grotesk'] font-bold uppercase tracking-wider text-neutral-700 pb-1 mb-2 border-b border-black flex items-center justify-between">
                <span>Medical Grounding Evidence</span>
                <span className="neo-badge neo-badge-green text-[8px]">NIH Verified</span>
              </div>

              <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {activeMetadata.retrieved_evidence.map((ev, i) => (
                  <div key={i} className="text-xs p-2 bg-neutral-50 border border-neutral-300 rounded">
                    <div className="font-bold text-[11px] text-blue-900 flex items-center justify-between">
                      <span>{ev.condition}</span>
                      <span className="text-[9px] font-mono text-neutral-500 uppercase">{ev.source_type}</span>
                    </div>
                    {ev.source_title && (
                      <div className="text-[10px] text-neutral-700 italic mt-0.5">
                        {ev.source_title}
                      </div>
                    )}
                    <div className="text-[10px] text-neutral-600 mt-1 leading-snug line-clamp-3">
                      {ev.excerpt}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="text-[10px] font-mono opacity-60 pt-3 border-t border-[var(--border-color)]">
          Dual-reward rejection logged on completion.
        </div>
      </div>

    </div>
  );
}
