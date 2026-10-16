import { getAuthToken } from "./api";
import { SSEMetadataEvent } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export interface StreamChatOptions {
  sessionId: string;
  queryText: string;
  selectedSymptoms?: string[];
  documentId?: string;
  onDiagnosticReport: (meta: SSEMetadataEvent) => void;
  onToken: (token: string) => void;
  onThinking?: (thinkingToken: string) => void;
  onDone: () => void;
  onError: (error: Error) => void;
}

export async function streamChatConsultation({
  sessionId,
  queryText,
  selectedSymptoms,
  documentId,
  onDiagnosticReport,
  onToken,
  onThinking,
  onDone,
  onError,
}: StreamChatOptions): Promise<() => void> {
  const controller = new AbortController();

  try {
    const token = await getAuthToken();
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        session_id: sessionId,
        query_text: queryText,
        selected_symptoms: selectedSymptoms || [],
        document_id: documentId || null,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `Stream request failed with HTTP ${response.status}`;
      try {
        const errJson = await response.json();
        detail = errJson.detail || detail;
      } catch {
        // ignore
      }
      throw new Error(detail);
    }

    if (!response.body) {
      throw new Error("No response body received for streaming.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    (async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split("\n\n");
          buffer = messages.pop() || "";

          for (const message of messages) {
            const lines = message.split("\n");
            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const payloadStr = line.slice(6).trim();
                if (payloadStr === "[DONE]") {
                  onDone();
                  return;
                }

                try {
                  const eventData = JSON.parse(payloadStr);
                  if (eventData.type === "diagnostic_report") {
                    onDiagnosticReport(eventData as SSEMetadataEvent);
                  } else if (eventData.type === "thinking") {
                    if (onThinking) {
                      onThinking(eventData.token || "");
                    }
                  } else if (eventData.type === "token") {
                    onToken(eventData.token || "");
                  }
                } catch {
                  // Non-json chunk or heartbeat
                }
              }
            }
          }
        }
        onDone();
      } catch (err: any) {
        if (err.name !== "AbortError") {
          onError(err);
        }
      }
    })();
  } catch (err: any) {
    onError(err);
  }

  // Return cancel function
  return () => {
    controller.abort();
  };
}

