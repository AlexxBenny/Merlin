import { useEffect, useState } from 'react'
import { Brain, RefreshCw, Database } from 'lucide-react'
import { api, type Memory } from '../lib/api'

const domainConfig: Record<string, { label: string; color: string; gradient: string }> = {
  preferences: { label: 'Preferences', color: '#a78bfa', gradient: 'linear-gradient(135deg, rgba(167,139,250,0.12), rgba(139,92,246,0.04))' },
  facts: { label: 'Facts', color: '#60a5fa', gradient: 'linear-gradient(135deg, rgba(96,165,250,0.12), rgba(59,130,246,0.04))' },
  traits: { label: 'Traits', color: '#34d399', gradient: 'linear-gradient(135deg, rgba(52,211,153,0.12), rgba(16,185,129,0.04))' },
  policies: { label: 'Policies', color: '#fbbf24', gradient: 'linear-gradient(135deg, rgba(251,191,36,0.12), rgba(245,158,11,0.04))' },
  relationships: { label: 'Relationships', color: '#f472b6', gradient: 'linear-gradient(135deg, rgba(244,114,182,0.12), rgba(236,72,153,0.04))' },
}

export default function MemoryPage() {
  const [memory, setMemory] = useState<Memory | null>(null)

  const load = () => api.getMemory().then(setMemory).catch(() => {})

  useEffect(() => {
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [])

  if (!memory) return (
    <div className="flex items-center justify-center h-[80vh]">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  const totalEntries = Object.values(domainConfig).reduce((sum, _, __, arr) => {
    const keys = Object.keys(domainConfig)
    return keys.reduce((s, k) => {
      const data = (memory as unknown as Record<string, Record<string, unknown>>)[k] || {}
      return s + (Array.isArray(data) ? data.length : Object.keys(data).length)
    }, 0)
  }, 0)

  return (
    <div className="page-enter space-y-6">
      <div className="section-header">
        <div>
          <h1 className="section-title">Memory</h1>
          <p className="section-subtitle">User knowledge store — 5 domains</p>
        </div>
        <button onClick={load} className="btn-ghost">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(domainConfig).map(([key, { label, color, gradient }]) => {
          const raw = (memory as unknown as Record<string, unknown>)[key] || {}
          const entries = Array.isArray(raw) ? raw.map((v, i) => [String(i), v] as [string, unknown]) : Object.entries(raw as Record<string, unknown>)

          return (
            <div key={key} className="glass-card overflow-hidden">
              {/* Header with gradient */}
              <div className="p-5 pb-4" style={{ background: gradient }}>
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                    style={{ background: color + '20', border: `1px solid ${color}30` }}>
                    <Brain size={16} style={{ color }} />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>{label}</h3>
                    <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{entries.length} entries</span>
                  </div>
                  {entries.length > 0 && (
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                      style={{ background: 'rgba(255,255,255,0.04)' }}>
                      <Database size={12} style={{ color: 'var(--color-text-muted)' }} />
                    </div>
                  )}
                </div>
              </div>

              {/* Entries */}
              <div className="px-5 pb-5 pt-3">
                {entries.length === 0 ? (
                  <p className="text-xs py-3 text-center" style={{ color: 'var(--color-text-muted)' }}>No data stored</p>
                ) : (
                  <div className="space-y-1.5 max-h-52 overflow-auto">
                    {entries.map(([k, v]) => {
                      const displayVal = typeof v === 'object' && v !== null
                        ? (v as Record<string, unknown>).value !== undefined
                          ? String((v as Record<string, unknown>).value)
                          : JSON.stringify(v)
                        : String(v)

                      return (
                        <div key={k} className="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors"
                          style={{ background: 'rgba(255,255,255,0.02)' }}
                          onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-bg-hover)')}
                          onMouseLeave={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}>
                          <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: color }} />
                          <span className="text-xs font-semibold font-mono" style={{ color }}>{k}</span>
                          <span className="text-xs ml-auto truncate max-w-[200px]" style={{ color: 'var(--color-text-secondary)' }}>
                            {displayVal}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
