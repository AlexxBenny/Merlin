import { useState, useEffect, useCallback } from 'react'
import { api } from '../lib/api'
import type { WhatsAppStatus, WhatsAppMessage } from '../lib/api'
import {
  MessageCircle, Send, RefreshCw, WifiOff,
  Clock, CheckCircle2, XCircle, User, Phone,
  AlertCircle,
} from 'lucide-react'

type Tab = 'messages' | 'compose'

const STATUS_CFG: Record<string, { cls: string; color: string; icon: typeof Clock; label: string }> = {
  sent:   { cls: 'badge-green', color: 'var(--emerald)', icon: CheckCircle2, label: 'Sent' },
  failed: { cls: 'badge-red',   color: 'var(--rose)',    icon: XCircle,      label: 'Failed' },
}

export default function WhatsApp() {
  const [tab, setTab] = useState<Tab>('messages')
  const [status, setStatus] = useState<WhatsAppStatus | null>(null)
  const [messages, setMessages] = useState<WhatsAppMessage[]>([])
  const [selected, setSelected] = useState<WhatsAppMessage | null>(null)
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState('')

  // Compose state
  const [compContact, setCompContact] = useState('')
  const [compText, setCompText] = useState('')
  const [sending, setSending] = useState(false)

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [s, m] = await Promise.all([
        api.getWhatsAppStatus(),
        api.getWhatsAppMessages(),
      ])
      setStatus(s)
      setMessages(m || [])
    } catch {
      setStatus(null)
      setMessages([])
    }
    setLoading(false)
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const flash = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 3000) }

  const handleSend = async () => {
    if (!compContact.trim() || !compText.trim()) { flash('Contact and message required'); return }
    setSending(true)
    try {
      await api.sendWhatsApp(compContact.trim(), compText.trim())
      flash('Message sent ✓')
      setCompContact(''); setCompText('')
      loadData()
    } catch (e) { flash(`Error: ${e}`) }
    setSending(false)
  }

  const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })

  const TABS: { key: Tab; icon: typeof MessageCircle; label: string; count: number }[] = [
    { key: 'messages', icon: MessageCircle, label: 'Messages', count: messages.length },
    { key: 'compose', icon: Send, label: 'Compose', count: 0 },
  ]

  const isConnected = status?.connected ?? false

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Toast */}
      {toast && (
        <div style={{ position: 'fixed', top: 20, right: 24, zIndex: 100, padding: '10px 18px', borderRadius: 10, background: 'var(--bg3)', border: '1px solid var(--border-accent)', color: 'var(--cyan)', fontSize: 13 }}>{toast}</div>
      )}

      {/* Status Bar + Tabs */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', gap: 4, padding: 4, background: 'var(--bg1)', borderRadius: 12, border: '1px solid var(--border)' }}>
          {TABS.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)} style={{
              display: 'flex', alignItems: 'center', gap: 7, padding: '7px 16px', borderRadius: 9,
              border: 'none', cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 12, transition: 'all 0.15s',
              background: tab === t.key ? 'var(--bg3)' : 'transparent',
              color: tab === t.key ? 'var(--cyan)' : 'var(--text-3)',
            }}>
              <t.icon size={13} />
              {t.label}
              {t.count > 0 && <span className="badge badge-cyan" style={{ padding: '1px 6px', fontSize: 10 }}>{t.count}</span>}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Connection status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {isConnected ? (
              <>
                <div className="ping-dot" style={{ background: 'var(--emerald)' }} />
                <span style={{ fontSize: 11, color: 'var(--emerald)' }}>Connected</span>
              </>
            ) : (
              <>
                <WifiOff size={12} style={{ color: 'var(--rose)' }} />
                <span style={{ fontSize: 11, color: 'var(--rose)' }}>Disconnected</span>
              </>
            )}
          </div>

          {/* Stats */}
          {status && (
            <span className="badge badge-muted" style={{ fontSize: 10 }}>
              {status.messages_sent_today} today · {status.rate_limit_remaining} remaining
            </span>
          )}

          <button className="btn" onClick={loadData}>
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </div>

      {/* Messages Tab */}
      {tab === 'messages' && (
        <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 14, minHeight: 400 }}>
          {/* Message List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {messages.length === 0 ? (
              <div className="card" style={{ textAlign: 'center', padding: '36px 0' }}>
                <MessageCircle size={24} style={{ color: 'var(--text-3)', margin: '0 auto 10px', display: 'block' }} />
                <span style={{ fontSize: 12, color: 'var(--text-3)' }}>No messages yet</span>
              </div>
            ) : messages.slice(0, 50).map(m => {
              const isSel = selected?.id === m.id
              const cfg = STATUS_CFG[m.status] || STATUS_CFG.sent
              return (
                <button key={m.id} onClick={() => setSelected(m)} style={{
                  textAlign: 'left', padding: '12px 14px', borderRadius: 11, cursor: 'pointer', fontFamily: 'var(--font-mono)',
                  background: isSel ? 'var(--cyan-glow)' : 'var(--bg2)', border: `1px solid ${isSel ? 'var(--border-accent)' : 'var(--border)'}`,
                  transition: 'all 0.15s',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                    <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                      {m.contact_name || m.recipient_id}
                    </span>
                    <span className={`badge ${cfg.cls}`} style={{ fontSize: 9, padding: '1px 6px', flexShrink: 0 }}>{cfg.label}</span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {m.content?.slice(0, 60) || '—'}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 3, opacity: 0.7 }}>
                    {fmtTime(m.timestamp)}
                  </div>
                </button>
              )
            })}
          </div>

          {/* Detail Panel */}
          <div className="card" style={{ padding: 22 }}>
            {selected ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Status */}
                {(() => { const c = STATUS_CFG[selected.status] || STATUS_CFG.sent; return (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`badge ${c.cls}`}>{c.label}</span>
                    <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{selected.id}</span>
                  </div>
                ) })()}

                {/* Contact */}
                <div>
                  <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Contact</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <User size={14} style={{ color: 'var(--cyan)', flexShrink: 0 }} />
                    <span style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-1)' }}>{selected.contact_name || 'Unknown'}</span>
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Phone</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <Phone size={14} style={{ color: 'var(--text-3)', flexShrink: 0 }} />
                    <span style={{ fontSize: 13, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{selected.recipient_id}</span>
                  </div>
                </div>

                {/* Message Content */}
                <div>
                  <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Message</label>
                  <div style={{
                    padding: '14px 16px', borderRadius: 12, background: 'var(--bg1)', border: '1px solid var(--border)',
                    fontSize: 13, color: 'var(--text-2)', lineHeight: 1.65, whiteSpace: 'pre-wrap', fontFamily: 'var(--font-mono)',
                  }}>
                    {selected.content || '—'}
                  </div>
                </div>

                {/* Timestamp */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-3)' }}>
                  <Clock size={12} />
                  {new Date(selected.timestamp * 1000).toLocaleString()}
                </div>

                {/* Error */}
                {selected.error && (
                  <div style={{ padding: '10px 14px', borderRadius: 9, background: 'rgba(244,63,94,0.06)', border: '1px solid rgba(244,63,94,0.20)', fontSize: 12 }}>
                    <span style={{ fontWeight: 500, color: 'var(--rose)' }}>Error: </span>
                    <span style={{ color: 'var(--text-3)' }}>{selected.error}</span>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 280, flexDirection: 'column', gap: 12 }}>
                <MessageCircle size={28} style={{ color: 'var(--text-3)' }} />
                <span style={{ fontSize: 13, color: 'var(--text-3)' }}>Select a message to view</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Compose Tab */}
      {tab === 'compose' && (
        <div className="card" style={{ padding: 22, maxWidth: 600 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18, padding: '10px 14px', background: 'var(--cyan-glow)', border: '1px solid var(--border-accent)', borderRadius: 9 }}>
            <AlertCircle size={13} style={{ color: 'var(--cyan)' }} />
            <span style={{ fontSize: 12, color: 'var(--text-3)' }}>For AI-generated messages, ask MERLIN in Chat. Use this form for quick manual sends.</span>
          </div>

          <div style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Contact</label>
            <input className="input" value={compContact} onChange={e => setCompContact(e.target.value)} placeholder="Name or phone number" disabled={sending} />
          </div>

          <div style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Message</label>
            <textarea className="input" rows={6} value={compText} onChange={e => setCompText(e.target.value)} placeholder="Write your message…" disabled={sending} />
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-green" onClick={handleSend} disabled={sending || !compContact.trim() || !compText.trim()}>
              <Send size={13} /> {sending ? 'Sending...' : 'Send Message'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
