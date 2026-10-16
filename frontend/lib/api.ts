import { auth } from "./firebase";
import { DiagnosticReportSchema, MemberProfile, SessionRecord, SymptomInputRequest } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function getAuthToken(forceRefresh: boolean = false): Promise<string> {
  // If running in browser, wait for Firebase Auth to restore session from storage
  if (typeof window !== "undefined") {
    if (typeof (auth as any).authStateReady === "function") {
      try {
        await auth.authStateReady();
      } catch (e) {
        console.warn("auth.authStateReady warning:", e);
      }
    }
  }

  const user = auth.currentUser;
  if (user) {
    try {
      const token = await user.getIdToken(forceRefresh);
      if (typeof window !== "undefined") {
        localStorage.setItem("healthtrack_last_uid", user.uid);
      }
      return token;
    } catch (e) {
      console.warn("Failed to retrieve Firebase ID token:", e);
    }
  }

  if (typeof window !== "undefined") {
    const demoToken = localStorage.getItem("healthtrack_demo_token");
    if (demoToken) return demoToken;
    const lastUid = localStorage.getItem("healthtrack_last_uid");
    if (lastUid && (lastUid.startsWith("patient_") || lastUid.startsWith("mem_") || lastUid.startsWith("dev_"))) {
      return `dev_${lastUid}`;
    }
  }
  return "dev_patient_alpha";
}

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  let token = await getAuthToken();
  const headers = new Headers(options.headers || {});
  
  if (!headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  
  if (!headers.has("Content-Type") && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const url = `${API_BASE}${path}`;
  let res = await fetch(url, {
    ...options,
    headers,
  });

  // Automatic token refresh retry on 401 if logged into Firebase
  if (res.status === 401 && auth.currentUser) {
    try {
      const freshToken = await getAuthToken(true);
      headers.set("Authorization", `Bearer ${freshToken}`);
      res = await fetch(url, {
        ...options,
        headers,
      });
    } catch {
      // Retry failed, fall through to error handling
    }
  }

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errorDetail;
    } catch {
      // Non-JSON response
    }
    throw new Error(errorDetail);
  }

  return res.json();
}

// ── Diagnostic Endpoints ──

export async function predictDisease(req: SymptomInputRequest): Promise<DiagnosticReportSchema> {
  return apiRequest<DiagnosticReportSchema>("/api/predict", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ── Family Members Endpoints ──

export async function listMembers(): Promise<{ status: string; members: MemberProfile[]; count: number }> {
  return apiRequest("/api/members");
}

export async function getMember(memberId: string): Promise<{ status: string; member: MemberProfile }> {
  return apiRequest(`/api/members/${memberId}`);
}

export async function createMember(member: Partial<MemberProfile>): Promise<{ status: string; member: MemberProfile }> {
  return apiRequest("/api/members", {
    method: "POST",
    body: JSON.stringify(member),
  });
}

export async function addMemberCondition(
  memberId: string,
  condition: { condition: string; icd10?: string; diagnosed_date?: string; status?: string; notes?: string }
): Promise<{ status: string; member: MemberProfile }> {
  return apiRequest(`/api/members/${memberId}/conditions`, {
    method: "POST",
    body: JSON.stringify(condition),
  });
}

// ── Session Endpoints ──

export async function createSession(data: { member_id?: string; title?: string; initial_symptoms?: string[] }): Promise<{ status: string; session: SessionRecord }> {
  return apiRequest("/api/sessions", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function listSessions(memberId?: string): Promise<{ status: string; sessions: SessionRecord[]; count: number }> {
  const query = memberId ? `?member_id=${encodeURIComponent(memberId)}` : "";
  return apiRequest(`/api/sessions${query}`);
}

export async function getSessionContext(sessionId: string): Promise<{ status: string; context: any }> {
  return apiRequest(`/api/sessions/${sessionId}`);
}

// ── File Upload ──

export async function uploadMedicalFile(file: File, documentType: string = "MEDICAL_DOCUMENT"): Promise<{ status: string; file: any }> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("document_type", documentType);

  return apiRequest("/api/upload", {
    method: "POST",
    body: formData,
  });
}

// ── System & Continual Ops ──

export async function getSystemHealth(): Promise<{ status: string; model_version: number; service: string }> {
  return apiRequest("/health");
}

export async function getSystemStatus(): Promise<any> {
  return apiRequest("/status");
}

export async function triggerContinualLearning(force: boolean = true): Promise<any> {
  return apiRequest("/trigger_learning", {
    method: "POST",
    body: JSON.stringify({ force }),
  });
}

export async function rollbackModel(): Promise<any> {
  return apiRequest("/models/rollback", {
    method: "POST",
  });
}

export async function listClinicalClasses(category?: string, tier?: string): Promise<{ status: string; total_classes: number; classes: any[] }> {
  const params = new URLSearchParams();
  if (category && category !== "All Systems") params.set("category", category);
  if (tier && tier !== "all") params.set("tier", tier);
  const q = params.toString() ? `?${params.toString()}` : "";
  return apiRequest(`/api/classes${q}`);
}

