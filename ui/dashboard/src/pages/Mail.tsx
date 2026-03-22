import { useState, useEffect, useCallback } from 'react'
import { api } from '../lib/api'
import type { Draft } from '../lib/api'
import {
  Mail as MailIcon, Send, Check, X, Copy,
  RefreshCw, FileEdit, PenSquare, Clock, Inbox,
  CheckCircle2, XCircle, AlertCircle,
} from 'lucide-react'

type Tab = 'drafts' | 'inbox' | 'compose'

const STATUS_CFG: Record<string, { cls: string; color: string; icon: typeof Clock; label: string }> = {
  pending_review: { cls: 'badge-amber', color: 'var(--amber)',  icon: Clock,        label: 'Pending Review' },
  approved:       { cls: 'badge-green', color: 'var(--emerald)', icon: CheckCircle2, label: 'Approved' },
  sent:           { cls: 'badge-cyan',  color: 'var(--cyan)',    icon: Send,         label: 'Sent' },
  discarded:      { cls: 'badge-red',   color: 'var(--rose)',    icon: XCircle,      label: 'Discarded' },
}

export default function Mail() {
  const [tab, setTab] = useState<Tab>('drafts')
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [selected, setSelected] = useState<Draft | null>(null)
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState('')
  const [compTo, setCompTo] = useState('')
  const [compSubject, setCompSubject] = useState('')
  const [compBody, setCompBody] = useState('')

  const loadDrafts = useCallback(async () => {
    setLoading(true)
    try { setDrafts(await api.getDrafts() || []) } catch { setDrafts([]) }
    setLoading(false)
  }, [])

  useEffect(() => { loadDrafts() }, [loadDrafts])

  const flash = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 3000) }

  const approve = async (id: string) => { try { await api.updateDraft(id, { status: 'approved' }); flash('Draft approved ✓'); loadDrafts(); if (selected?.id === id) setSelected({ ...selected, status: 'approved' }) } catch (e) { flash(`Error: ${e}`) } }
  const sendDraft = async (id: string) => { try { await api.sendDraft(id); flash('Email sent ✓'); loadDrafts(); setSelected(null) } catch (e) { flash(`Error: ${e}`) } }
  const discard = async (id: string) => { try { await api.deleteDraft(id); flash('Draft discarded'); loadDrafts(); if (selected?.id === id) setSelected(null) } catch (e) { flash(`Error: ${e}`) } }
  const copyDraft = (d: Draft) => { navigator.clipboard.writeText(`To: ${d.recipient}\nSubject: ${d.subject}\n\n${d.body}`); flash('Copied ✓') }
  const saveDraft = async () => { if (!selected) return; try { await api.updateDraft(selected.id, { recipient: selected.recipient, subject: selected.subject, body: selected.body }); flash('Saved ✓'); loadDrafts() } catch (e) { flash(`Error: ${e}`) } }

  const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

  const active = drafts.filter(d => d.status !== 'discarded' && d.status !== 'sent')
  const sent = drafts.filter(d => d.status === 'sent')

  const TABS: { key: Tab; icon: typeof FileEdit; label: string; count: number }[] = [
    { key: 'drafts', icon: FileEdit, label: 'Drafts', count: active.length },
    { key: 'inbox', icon: Inbox, label: 'Inbox', count: 0 },
    { key: 'compose', icon: PenSquare, label: 'Compose', count: 0 },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Toast */}
      {toast && (
        <div style={{ position: 'fixed', top: 20, right: 24, zIndex: 100, padding: '10px 18px', borderRadius: 10, background: 'var(--bg3)', border: '1px solid var(--border-accent)', color: 'var(--cyan)', fontSize: 13 }}>{toast}</div>
      )}

      {/* Actions + Tabs */}
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
        <button className="btn" onClick={loadDrafts}><RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh</button>
      </div>

      {/* Drafts Tab */}
      {tab === 'drafts' && (
        <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', gap: 14, minHeight: 400 }}>
          {/* List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {active.length === 0 ? (
              <div className="card" style={{ textAlign: 'center', padding: '36px 0' }}>
                <MailIcon size={24} style={{ color: 'var(--text-3)', margin: '0 auto 10px', display: 'block' }} />
                <span style={{ fontSize: 12, color: 'var(--text-3)' }}>No drafts yet</span>
              </div>
            ) : active.map(d => {
              const isSel = selected?.id === d.id
              const cfg = STATUS_CFG[d.status] || STATUS_CFG.pending_review
              return (
                <button key={d.id} onClick={() => setSelected(d)} style={{
                  textAlign: 'left', padding: '12px 14px', borderRadius: 11, cursor: 'pointer', fontFamily: 'var(--font-mono)',
                  background: isSel ? 'var(--cyan-glow)' : 'var(--bg2)', border: `1px solid ${isSel ? 'var(--border-accent)' : 'var(--border)'}`,
                  transition: 'all 0.15s',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                    <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{d.subject || 'No Subject'}</span>
                    <span className={`badge ${cfg.cls}`} style={{ fontSize: 9, padding: '1px 6px', flexShrink: 0 }}>{cfg.label}</span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>To: {d.recipient}</div>
                </button>
              )
            })}
            {sent.length > 0 && (
              <>
                <div style={{ padding: '10px 4px 4px', fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.10em', color: 'var(--text-3)' }}>Sent</div>
                {sent.slice(0, 5).map(d => (
                  <div key={d.id} style={{ padding: '10px 14px', borderRadius: 11, background: 'var(--bg2)', border: '1px solid var(--border)', opacity: 0.6 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.subject || 'No Subject'}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 3 }}>To: {d.recipient} · {fmtTime(d.updated_at)}</div>
                  </div>
                ))}
              </>
            )}
          </div>

          {/* Detail */}
          <div className="card" style={{ padding: 22 }}>
            {selected ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {/* Status */}
                {(() => { const c = STATUS_CFG[selected.status] || STATUS_CFG.pending_review; return (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`badge ${c.cls}`}>{c.label}</span>
                    <span style={{ fontSize: 10, color: 'var(--text-3)' }}>{selected.id}</span>
                  </div>
                ) })()}

                {/* Fields */}
                {(['recipient', 'subject'] as const).map(f => (
                  <div key={f}>
                    <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{f === 'recipient' ? 'To' : 'Subject'}</label>
                    <input className="input" value={selected[f]} onChange={e => setSelected({ ...selected, [f]: e.target.value })} />
                  </div>
                ))}
                <div>
                  <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Body</label>
                  <textarea className="input" rows={9} value={selected.body} onChange={e => setSelected({ ...selected, body: e.target.value })} />
                </div>

                {selected.source_query && (
                  <div style={{ padding: '10px 14px', borderRadius: 9, background: 'var(--cyan-glow)', border: '1px solid var(--border-accent)', fontSize: 12, color: 'var(--text-3)' }}>
                    <span style={{ fontWeight: 500, color: 'var(--cyan)' }}>Origin: </span>"{selected.source_query}"
                  </div>
                )}

                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button className="btn" onClick={saveDraft}><FileEdit size={13} /> Save</button>
                  {selected.status === 'pending_review' && <button className="btn btn-green" onClick={() => approve(selected.id)}><Check size={13} /> Approve</button>}
                  {selected.status === 'approved' && <button className="btn btn-cyan" onClick={() => sendDraft(selected.id)}><Send size={13} /> Send</button>}
                  <button className="btn" onClick={() => copyDraft(selected)}><Copy size={13} /> Copy</button>
                  <button className="btn btn-red" style={{ marginLeft: 'auto' }} onClick={() => discard(selected.id)}><X size={13} /> Discard</button>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 280, flexDirection: 'column', gap: 12 }}>
                <MailIcon size={28} style={{ color: 'var(--text-3)' }} />
                <span style={{ fontSize: 13, color: 'var(--text-3)' }}>Select a draft to review</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Inbox Tab */}
      {tab === 'inbox' && (
        <div className="card" style={{ padding: 48, textAlign: 'center' }}>
          <Inbox size={32} style={{ color: 'var(--text-3)', margin: '0 auto 10px', display: 'block' }} />
          <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-2)' }}>Inbox coming soon</p>
          <p style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 4 }}>Ask MERLIN to check your inbox via chat</p>
        </div>
      )}

      {/* Compose Tab */}
      {tab === 'compose' && (
        <div className="card" style={{ padding: 22, maxWidth: 600 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18, padding: '10px 14px', background: 'var(--cyan-glow)', border: '1px solid var(--border-accent)', borderRadius: 9 }}>
            <AlertCircle size={13} style={{ color: 'var(--cyan)' }} />
            <span style={{ fontSize: 12, color: 'var(--text-3)' }}>For AI-generated emails, ask MERLIN in Chat. Use this form for quick manual drafts.</span>
          </div>
          {[{ l: 'To', v: compTo, s: setCompTo, t: 'email' }, { l: 'Subject', v: compSubject, s: setCompSubject, t: 'text' }].map(f => (
            <div key={f.l} style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{f.l}</label>
              <input className="input" value={f.v} onChange={e => f.s(e.target.value)} placeholder={f.l === 'To' ? 'recipient@example.com' : 'Subject'} />
            </div>
          ))}
          <div style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Body</label>
            <textarea className="input" rows={8} value={compBody} onChange={e => setCompBody(e.target.value)} placeholder="Write your message…" />
          </div>
          <button className="btn btn-cyan"><Send size={13} /> Save Draft</button>
        </div>
      )}
    </div>
  )
}
