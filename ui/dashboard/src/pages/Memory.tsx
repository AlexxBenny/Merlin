import { useEffect, useState } from 'react'
import { Brain } from 'lucide-react'
import { api, type Memory } from '../lib/api'

const domainLabels: Record<string, { label: string; color: string }> = {
  preferences: { label: 'Preferences', color: '#a78bfa' },
  facts: { label: 'Facts', color: '#3b82f6' },
  traits: { label: 'Traits', color: '#22c55e' },
  policies: { label: 'Policies', color: '#f59e0b' },
  relationships: { label: 'Relationships', color: '#ec4899' },
}

export default function MemoryPage() {
  const [memory, setMemory] = useState<Memory | null>(null)

  useEffect(() => {
    api.getMemory().then(setMemory).catch(() => {})
  }, [])

  if (!memory) return (
    <div className="flex items-center justify-center h-full">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  return (
    <div className="page-enter space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>Memory</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
          User knowledge store — 5 domains
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(domainLabels).map(([key, { label, color }]) => {
          const data = (memory as unknown as Record<string, Record<string, unknown>>)[key] || {}
          const entries = Object.entries(data)

          return (
            <div key={key} className="glass-card p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: color + '20' }}>
                  <Brain size={16} style={{ color }} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>{label}</h3>
                  <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{entries.length} entries</span>
                </div>
              </div>

              {entries.length === 0 ? (
                <p className="text-xs italic" style={{ color: 'var(--color-text-muted)' }}>No data stored</p>
              ) : (
                <div className="space-y-2 max-h-60 overflow-auto">
                  {entries.map(([k, v]) => (
                    <div key={k} className="px-3 py-2 rounded-lg text-xs" style={{ background: 'var(--color-bg-tertiary)' }}>
                      <span className="font-medium" style={{ color }}>{k}:</span>{' '}
                      <span style={{ color: 'var(--color-text-secondary)' }}>
                        {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
