// src/lib/api.ts — Typed API client for MERLIN backend

const API_BASE = '/api/v1';
const WS_BASE = `ws://${window.location.host}`;

// ── Types ───────────────────────────────────────────────

export interface SystemState {
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  uptime_seconds: number;
  mission_state: string;
  battery_percent?: number;
  battery_charging?: boolean;
  timestamp: number;
}

export interface Job {
  id: string;
  type: string;
  query: string;
  status: string;
  priority: string;
  next_run: number | null;
  attempts: number;
  max_retries: number;
  created_at: number;
  error?: string;
  schedule?: Record<string, unknown>;
}

export interface Memory {
  preferences: Record<string, unknown>;
  facts: Record<string, unknown>;
  traits: Record<string, unknown>;
  policies: Record<string, unknown>;
  relationships: Record<string, unknown>;
}

export interface Mission {
  mission_id: string;
  timestamp: number;
  nodes_executed: string[];
  nodes_skipped: string[];
  nodes_failed: string[];
  nodes_timed_out: string[];
  active_entity?: string;
  active_domain?: string;
  recovery_attempted: boolean;
  plan?: {
    id: string;
    nodes: MissionNode[];
    metadata: Record<string, unknown>;
  };
  node_statuses?: Record<string, string>;
  query?: string;
}

export interface MissionNode {
  id: string;
  skill: string;
  inputs: Record<string, unknown>;
  depends_on: string[];
  mode: string;
  outputs?: Record<string, unknown>;
}

export interface LogEntry {
  timestamp: number;
  level: string;
  module: string;
  logger: string;
  message: string;
  lineno: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface ChatResponse {
  id: string;
  status: string;
  response: string;
}

export interface Draft {
  id: string;
  recipient: string;
  cc: string;
  bcc: string;
  subject: string;
  body: string;
  status: 'pending_review' | 'approved' | 'sent' | 'discarded';
  attachments: { path: string; filename: string; mime_type: string }[];
  source_query: string;
  intent_source: string;
  reply_to_message_id?: string;
  thread_id?: string;
  created_at: number;
  updated_at: number;
}

export interface WhatsAppStatus {
  connected: boolean;
  messages_sent_today: number;
  total_messages: number;
  rate_limit_remaining: number;
}

export interface WhatsAppMessage {
  id: string;
  channel: string;
  recipient_id: string;
  contact_name: string;
  direction: string;
  content_type: string;
  content: string;
  status: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
  error?: string;
}

export interface DraftUpdate {
  recipient?: string;
  cc?: string;
  bcc?: string;
  subject?: string;
  body?: string;
  status?: string;
}

// ── Fetch helpers ───────────────────────────────────────

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

// ── API functions ───────────────────────────────────────

export const api = {
  getSystem: () => get<SystemState>('/system'),
  getJobs: () => get<Job[]>('/jobs'),
  cancelJob: (id: string) => del<{ status: string; message: string }>(`/jobs/${id}`),
  pauseJob: (id: string) => patch<{ status: string; message: string }>(`/jobs/${id}`, { action: 'pause' }),
  resumeJob: (id: string) => patch<{ status: string; message: string }>(`/jobs/${id}`, { action: 'resume' }),
  getMemory: () => get<Memory>('/memory'),
  getMissions: () => get<Mission[]>('/missions'),
  getMission: (id: string) => get<Mission>(`/missions/${id}`),
  getWorld: () => get<Record<string, unknown>>('/world'),
  getConfig: () => get<Record<string, unknown>>('/config'),
  updateConfig: (updates: Record<string, unknown>) => patch<{ status: string; message: string }>('/config', { updates }),
  getLogs: (n = 200, level?: string) => get<LogEntry[]>(`/logs?n=${n}${level ? `&level=${level}` : ''}`),
  chat: (message: string) => post<ChatResponse>('/chat', { message }),
  getChatHistory: () => get<{ messages: ChatMessage[] }>('/chat/history'),
  newChatSession: () => post<{ status: string }>('/chat/new_session', {}),
  getHealth: () => get<{ status: string; merlin_connected: boolean }>('/health'),

  // ── Drafts (email integration) ──────────────────────────
  getDrafts: () => get<Draft[]>('/drafts'),
  getDraft: (id: string) => get<Draft>(`/drafts/${id}`),
  updateDraft: (id: string, updates: DraftUpdate) => patch<{ status: string; response: string }>(`/drafts/${id}`, updates),
  deleteDraft: (id: string) => del<{ status: string; response: string }>(`/drafts/${id}`),
  sendDraft: (id: string) => post<{ status: string; response: string }>(`/drafts/${id}/send`, {}),

  // ── WhatsApp ─────────────────────────────────────────────
  getWhatsAppStatus: () => get<WhatsAppStatus>('/whatsapp/status'),
  getWhatsAppMessages: () => get<WhatsAppMessage[]>('/whatsapp/messages'),
  sendWhatsApp: (contact: string, text: string) => post<{ status: string; response: string }>('/whatsapp/send', { contact, text }),
};

// ── WebSocket helper ────────────────────────────────────

export function createWebSocket(
  path: string,
  onMessage: (data: unknown) => void,
  onError?: (err: Event) => void,
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}${path}`);
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      onMessage(event.data);
    }
  };
  ws.onerror = onError || (() => {});
  return ws;
}

// ── SSE helper for chat streaming ───────────────────────

export async function streamChat(
  message: string,
  onChunk: (text: string) => void,
  onDone: (fullResponse: string) => void,
  onError?: (err: string) => void,
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    if (!res.ok || !res.body) {
      onError?.(`API error: ${res.status}`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'chunk') onChunk(data.text);
            if (data.type === 'done') onDone(data.full_response);
          } catch {}
        }
      }
    }
  } catch (e) {
    onError?.(String(e));
  }
}
