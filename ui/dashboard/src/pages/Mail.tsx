// src/pages/Mail.tsx — Email integration Mail page

import { useState, useEffect, useCallback } from 'react'
import { api } from '../lib/api'
import type { Draft } from '../lib/api'
import {
  Mail as MailIcon, Send, Check, X, Copy,
  RefreshCw, Inbox, FileEdit, PenSquare, Clock,
  CheckCircle2, XCircle, AlertCircle,
} from 'lucide-react'

type Tab = 'drafts' | 'inbox' | 'compose'

const STATUS_CONFIG: Record<string, { color: string; icon: typeof Clock; label: string }> = {
  pending_review: { color: '#f59e0b', icon: Clock, label: 'Pending Review' },
  approved: { color: '#22c55e', icon: CheckCircle2, label: 'Approved' },
  sent: { color: '#00d4ff', icon: Send, label: 'Sent' },
  discarded: { color: '#ef4444', icon: XCircle, label: 'Discarded' },
}

export default function Mail() {
  const [tab, setTab] = useState<Tab>('drafts')
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [selectedDraft, setSelectedDraft] = useState<Draft | null>(null)
  const [loading, setLoading] = useState(false)
  const [actionMsg, setActionMsg] = useState('')

  // Compose state
  const [compTo, setCompTo] = useState('')
  const [compSubject, setCompSubject] = useState('')
  const [compBody, setCompBody] = useState('')

  const loadDrafts = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getDrafts()
      setDrafts(data || [])
    } catch {
      setDrafts([])
    }
    setLoading(false)
  }, [])

  useEffect(() => { loadDrafts() }, [loadDrafts])

  const showAction = (msg: string) => {
    setActionMsg(msg)
    setTimeout(() => setActionMsg(''), 3000)
  }

  const handleApprove = async (id: string) => {
    try {
      await api.updateDraft(id, { status: 'approved' })
      showAction('Draft approved ✓')
      loadDrafts()
      if (selectedDraft?.id === id) {
        setSelectedDraft({ ...selectedDraft, status: 'approved' })
      }
    } catch (e) { showAction(`Error: ${e}`) }
  }

  const handleSend = async (id: string) => {
    try {
      await api.sendDraft(id)
      showAction('Email sent ✓')
      loadDrafts()
      setSelectedDraft(null)
    } catch (e) { showAction(`Error: ${e}`) }
  }

  const handleDiscard = async (id: string) => {
    try {
      await api.deleteDraft(id)
      showAction('Draft discarded')
      loadDrafts()
      if (selectedDraft?.id === id) setSelectedDraft(null)
    } catch (e) { showAction(`Error: ${e}`) }
  }

  const handleCopy = (draft: Draft) => {
    const text = `To: ${draft.recipient}\nSubject: ${draft.subject}\n\n${draft.body}`
    navigator.clipboard.writeText(text)
    showAction('Copied to clipboard ✓')
  }

  const handleSaveDraft = async () => {
    if (selectedDraft) {
      try {
        await api.updateDraft(selectedDraft.id, {
          recipient: selectedDraft.recipient,
          subject: selectedDraft.subject,
          body: selectedDraft.body,
        })
        showAction('Draft saved ✓')
        loadDrafts()
      } catch (e) { showAction(`Error: ${e}`) }
    }
  }

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000)
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  }

  const activeDrafts = drafts.filter(d => d.status !== 'discarded' && d.status !== 'sent')
  const sentDrafts = drafts.filter(d => d.status === 'sent')

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <MailIcon size={22} style={{ color: 'var(--color-accent)' }} />
          <h1 className="text-xl font-bold" style={{ color: 'var(--color-text-primary)' }}>
            Mail
          </h1>
        </div>
        <button
          onClick={loadDrafts}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
          style={{
            background: 'var(--color-bg-hover)',
            color: 'var(--color-text-secondary)',
            border: '1px solid var(--color-border)',
          }}
        >
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Action toast */}
      {actionMsg && (
        <div className="fixed top-4 right-4 z-50 px-4 py-2 rounded-lg text-sm font-medium animate-pulse"
          style={{
            background: 'linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,153,204,0.1))',
            border: '1px solid var(--color-accent)',
            color: 'var(--color-accent)',
          }}>
          {actionMsg}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-6 p-1 rounded-xl" style={{ background: 'var(--color-bg-secondary)' }}>
        {([
          { key: 'drafts' as Tab, icon: FileEdit, label: 'Drafts', count: activeDrafts.length },
          { key: 'inbox' as Tab, icon: Inbox, label: 'Inbox', count: 0 },
          { key: 'compose' as Tab, icon: PenSquare, label: 'Compose', count: 0 },
        ]).map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all"
            style={{
              background: tab === t.key ? 'var(--color-bg-hover)' : 'transparent',
              color: tab === t.key ? 'var(--color-accent)' : 'var(--color-text-muted)',
              border: tab === t.key ? '1px solid var(--color-border)' : '1px solid transparent',
            }}
          >
            <t.icon size={14} />
            {t.label}
            {t.count > 0 && (
              <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold"
                style={{ background: 'rgba(0,212,255,0.15)', color: 'var(--color-accent)' }}>
                {t.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Drafts Tab */}
      {tab === 'drafts' && (
        <div className="flex gap-4" style={{ minHeight: '400px' }}>
          {/* Draft list */}
          <div className="w-80 shrink-0 space-y-2">
            {activeDrafts.length === 0 ? (
              <div className="text-center py-12 rounded-xl" style={{ background: 'var(--color-bg-secondary)', color: 'var(--color-text-muted)' }}>
                <MailIcon size={32} className="mx-auto mb-3 opacity-30" />
                <p className="text-sm">No drafts yet</p>
                <p className="text-xs mt-1">Ask MERLIN to compose an email</p>
              </div>
            ) : (
              activeDrafts.map(draft => {
                const conf = STATUS_CONFIG[draft.status] || STATUS_CONFIG.pending_review
                const Icon = conf.icon
                return (
                  <div
                    key={draft.id}
                    onClick={() => setSelectedDraft(draft)}
                    className="p-3 rounded-xl cursor-pointer transition-all"
                    style={{
                      background: selectedDraft?.id === draft.id
                        ? 'rgba(0,212,255,0.08)'
                        : 'var(--color-bg-secondary)',
                      border: selectedDraft?.id === draft.id
                        ? '1px solid var(--color-accent)'
                        : '1px solid var(--color-border)',
                    }}
                  >
                    <div className="flex items-start justify-between mb-1">
                      <span className="text-xs font-semibold truncate flex-1"
                        style={{ color: 'var(--color-text-primary)' }}>
                        {draft.subject || 'No Subject'}
                      </span>
                      <Icon size={12} style={{ color: conf.color, marginLeft: 8, flexShrink: 0 }} />
                    </div>
                    <div className="text-[11px] truncate" style={{ color: 'var(--color-text-muted)' }}>
                      To: {draft.recipient}
                    </div>
                    <div className="text-[10px] mt-1" style={{ color: 'var(--color-text-muted)' }}>
                      {formatTime(draft.created_at)}
                    </div>
                  </div>
                )
              })
            )}

            {sentDrafts.length > 0 && (
              <>
                <div className="pt-4 pb-1 px-1">
                  <span className="text-[10px] font-semibold tracking-widest uppercase"
                    style={{ color: 'var(--color-text-muted)' }}>Sent</span>
                </div>
                {sentDrafts.slice(0, 5).map(draft => (
                  <div key={draft.id} className="p-3 rounded-xl opacity-60"
                    style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}>
                    <div className="text-xs font-medium truncate" style={{ color: 'var(--color-text-secondary)' }}>
                      {draft.subject || 'No Subject'}
                    </div>
                    <div className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                      To: {draft.recipient} · {formatTime(draft.updated_at)}
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>

          {/* Draft detail / editor */}
          <div className="flex-1 rounded-xl p-5" style={{
            background: 'var(--color-bg-secondary)',
            border: '1px solid var(--color-border)',
          }}>
            {selectedDraft ? (
              <>
                {/* Status badge */}
                {(() => {
                  const conf = STATUS_CONFIG[selectedDraft.status] || STATUS_CONFIG.pending_review
                  return (
                    <div className="flex items-center gap-2 mb-4">
                      <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold"
                        style={{ background: `${conf.color}20`, color: conf.color }}>
                        {conf.label}
                      </span>
                      <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                        {selectedDraft.id}
                      </span>
                    </div>
                  )
                })()}

                {/* Fields */}
                <div className="space-y-3 mb-4">
                  <div>
                    <label className="text-[10px] font-semibold tracking-wider uppercase mb-1 block"
                      style={{ color: 'var(--color-text-muted)' }}>To</label>
                    <input
                      value={selectedDraft.recipient}
                      onChange={e => setSelectedDraft({ ...selectedDraft, recipient: e.target.value })}
                      className="w-full px-3 py-2 rounded-lg text-sm"
                      style={{
                        background: 'var(--color-bg-primary)',
                        color: 'var(--color-text-primary)',
                        border: '1px solid var(--color-border)',
                      }}
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-semibold tracking-wider uppercase mb-1 block"
                      style={{ color: 'var(--color-text-muted)' }}>Subject</label>
                    <input
                      value={selectedDraft.subject}
                      onChange={e => setSelectedDraft({ ...selectedDraft, subject: e.target.value })}
                      className="w-full px-3 py-2 rounded-lg text-sm"
                      style={{
                        background: 'var(--color-bg-primary)',
                        color: 'var(--color-text-primary)',
                        border: '1px solid var(--color-border)',
                      }}
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-semibold tracking-wider uppercase mb-1 block"
                      style={{ color: 'var(--color-text-muted)' }}>Body</label>
                    <textarea
                      value={selectedDraft.body}
                      onChange={e => setSelectedDraft({ ...selectedDraft, body: e.target.value })}
                      rows={12}
                      className="w-full px-3 py-2 rounded-lg text-sm resize-y"
                      style={{
                        background: 'var(--color-bg-primary)',
                        color: 'var(--color-text-primary)',
                        border: '1px solid var(--color-border)',
                        fontFamily: 'inherit',
                        lineHeight: '1.6',
                      }}
                    />
                  </div>
                </div>

                {/* Source query */}
                {selectedDraft.source_query && (
                  <div className="mb-4 p-3 rounded-lg text-[11px]" style={{
                    background: 'rgba(0,212,255,0.05)',
                    border: '1px solid rgba(0,212,255,0.1)',
                    color: 'var(--color-text-muted)',
                  }}>
                    <span className="font-semibold" style={{ color: 'var(--color-accent)' }}>Origin:</span>{' '}
                    "{selectedDraft.source_query}"
                  </div>
                )}

                {/* Action buttons */}
                <div className="flex gap-2 flex-wrap">
                  <button onClick={handleSaveDraft}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80"
                    style={{ background: 'var(--color-bg-hover)', color: 'var(--color-text-primary)', border: '1px solid var(--color-border)' }}>
                    <FileEdit size={13} /> Save
                  </button>
                  {selectedDraft.status === 'pending_review' && (
                    <button onClick={() => handleApprove(selectedDraft.id)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80"
                      style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.3)' }}>
                      <Check size={13} /> Approve
                    </button>
                  )}
                  {selectedDraft.status === 'approved' && (
                    <button onClick={() => handleSend(selectedDraft.id)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,153,204,0.15))',
                        color: 'var(--color-accent)',
                        border: '1px solid rgba(0,212,255,0.3)',
                      }}>
                      <Send size={13} /> Send
                    </button>
                  )}
                  <button onClick={() => handleCopy(selectedDraft)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80"
                    style={{ background: 'var(--color-bg-hover)', color: 'var(--color-text-secondary)', border: '1px solid var(--color-border)' }}>
                    <Copy size={13} /> Copy
                  </button>
                  <button onClick={() => handleDiscard(selectedDraft.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80 ml-auto"
                    style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.2)' }}>
                    <X size={13} /> Discard
                  </button>
                </div>
              </>
            ) : (
              <div className="h-full flex items-center justify-center" style={{ color: 'var(--color-text-muted)' }}>
                <div className="text-center">
                  <MailIcon size={40} className="mx-auto mb-3 opacity-20" />
                  <p className="text-sm">Select a draft to review</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Inbox Tab */}
      {tab === 'inbox' && (
        <div className="rounded-xl p-8 text-center" style={{
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
        }}>
          <Inbox size={40} className="mx-auto mb-3 opacity-20" style={{ color: 'var(--color-text-muted)' }} />
          <p className="text-sm font-medium" style={{ color: 'var(--color-text-secondary)' }}>
            Inbox coming soon
          </p>
          <p className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
            Ask MERLIN to check your inbox via chat
          </p>
        </div>
      )}

      {/* Compose Tab */}
      {tab === 'compose' && (
        <div className="rounded-xl p-5" style={{
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
        }}>
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle size={14} style={{ color: 'var(--color-accent)' }} />
            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              For AI-generated emails, ask MERLIN in Chat. Use this form for quick manual drafts.
            </p>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-[10px] font-semibold tracking-wider uppercase mb-1 block"
                style={{ color: 'var(--color-text-muted)' }}>To</label>
              <input
                value={compTo}
                onChange={e => setCompTo(e.target.value)}
                placeholder="recipient@example.com"
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{
                  background: 'var(--color-bg-primary)',
                  color: 'var(--color-text-primary)',
                  border: '1px solid var(--color-border)',
                }}
              />
            </div>
            <div>
              <label className="text-[10px] font-semibold tracking-wider uppercase mb-1 block"
                style={{ color: 'var(--color-text-muted)' }}>Subject</label>
              <input
                value={compSubject}
                onChange={e => setCompSubject(e.target.value)}
                placeholder="Subject"
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{
                  background: 'var(--color-bg-primary)',
                  color: 'var(--color-text-primary)',
                  border: '1px solid var(--color-border)',
                }}
              />
            </div>
            <div>
              <label className="text-[10px] font-semibold tracking-wider uppercase mb-1 block"
                style={{ color: 'var(--color-text-muted)' }}>Body</label>
              <textarea
                value={compBody}
                onChange={e => setCompBody(e.target.value)}
                rows={8}
                placeholder="Write your message..."
                className="w-full px-3 py-2 rounded-lg text-sm resize-y"
                style={{
                  background: 'var(--color-bg-primary)',
                  color: 'var(--color-text-primary)',
                  border: '1px solid var(--color-border)',
                  fontFamily: 'inherit',
                  lineHeight: '1.6',
                }}
              />
            </div>
          </div>

          <div className="mt-4">
            <p className="text-[11px] italic" style={{ color: 'var(--color-text-muted)' }}>
              Manual compose creates a draft — it will still require approval before sending.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
