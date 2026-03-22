import { useEffect, useState } from 'react'
import { Brain, RefreshCw, Database } from 'lucide-react'
import { api, type Memory } from '../lib/api'

const DOMAINS: { key: string; label: string; color: string }[] = [
  { key: 'preferences',   label: 'Preferences',   color: '#8b5cf6' },
  { key: 'facts',         label: 'Facts',          color: '#60a5fa' },
  { key: 'traits',        label: 'Traits',         color: '#10b981' },
  { key: 'policies',      label: 'Policies',       color: '#f59e0b' },
  { key: 'relationships', label: 'Relationships',  color: '#f472b6' },
]

export default function MemoryPage() {
  const [memory, setMemory] = useState<Memory | null>(null)
  const load = () => api.getMemory().then(setMemory).catch(() => {})

  useEffect(() => { load(); const i = setInterval(load, 5000); return () => clearInterval(i) }, [])

  if (!memory) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '70vh' }}>
      <div className="animate-bop" style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--cyan)' }} />
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button className="btn" onClick={load}><RefreshCw size={13} /> Refresh</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
        {DOMAINS.map(({ key, label, color }) => {
          const raw = (memory as unknown as Record<string, unknown>)[key] || {}
          const entries = Array.isArray(raw) ? raw.map((v, i) => [String(i), v] as [string, unknown]) : Object.entries(raw as Record<string, unknown>)

          return (
            <div key={key} className="card" style={{ overflow: 'hidden' }}>
              {/* Domain header */}
              <div style={{ padding: '16px 18px 14px', background: `${color}08`, borderBottom: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 34, height: 34, borderRadius: 9, background: `${color}18`, border: `1px solid ${color}28`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Database size={14} style={{ color }} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>{label}</h3>
                    <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{entries.length} entries</span>
                  </div>
                </div>
              </div>

              {/* Entries */}
              <div style={{ padding: '12px 16px', maxHeight: 208, overflowY: 'auto' }}>
                {entries.length === 0 ? (
                  <p style={{ fontSize: 12, color: 'var(--text-3)', textAlign: 'center', padding: '12px 0' }}>No data stored</p>
                ) : (
                  entries.map(([k, v]) => {
                    const val = typeof v === 'object' && v !== null
                      ? (v as Record<string, unknown>).value !== undefined ? String((v as Record<string, unknown>).value) : JSON.stringify(v)
                      : String(v)
                    return (
                      <div key={k} style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        padding: '8px 10px', borderRadius: 8, marginBottom: 4,
                        background: 'rgba(255,255,255,0.015)', transition: 'background 0.12s', cursor: 'default',
                      }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.04)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.015)')}
                      >
                        <span style={{ width: 5, height: 5, borderRadius: '50%', background: color, flexShrink: 0 }} />
                        <span style={{ fontSize: 11, fontWeight: 500, color, minWidth: 100 }}>{k}</span>
                        <span style={{ fontSize: 12, color: 'var(--text-2)', marginLeft: 'auto', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 200 }}>{val}</span>
                      </div>
                    )
                  })
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
