import { useEffect, useState } from 'react'
import { Globe, ChevronRight, ChevronDown, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'

function TreeNode({ label, value, depth = 0 }: { label: string; value: unknown; depth?: number }) {
  const [open, setOpen] = useState(depth < 2)
  const isObj = typeof value === 'object' && value !== null && !Array.isArray(value)
  const isArr = Array.isArray(value)
  const isExpandable = isObj || isArr

  return (
    <div style={{ paddingLeft: depth * 16 }}>
      <div
        className="flex items-center gap-1 py-1 px-2 rounded cursor-pointer hover:bg-[var(--color-bg-hover)] transition-colors"
        onClick={() => isExpandable && setOpen(!open)}
      >
        {isExpandable ? (
          open ? <ChevronDown size={12} style={{ color: 'var(--color-text-muted)' }} />
               : <ChevronRight size={12} style={{ color: 'var(--color-text-muted)' }} />
        ) : (
          <span className="w-3" />
        )}

        <span className="text-xs font-medium" style={{ color: 'var(--color-accent-dim)' }}>
          {label}
        </span>

        {!isExpandable && (
          <span className="text-xs ml-2" style={{
            color: value === null || value === undefined ? 'var(--color-text-muted)' :
              typeof value === 'boolean' ? (value ? 'var(--color-success)' : 'var(--color-error)') :
              typeof value === 'number' ? 'var(--color-warning)' : 'var(--color-text-secondary)'
          }}>
            {value === null ? 'null' : value === undefined ? 'undefined' : String(value)}
          </span>
        )}

        {isArr && (
          <span className="text-xs ml-1" style={{ color: 'var(--color-text-muted)' }}>
            [{(value as unknown[]).length}]
          </span>
        )}
      </div>

      {open && isObj && (
        Object.entries(value as Record<string, unknown>).map(([k, v]) => (
          <TreeNode key={k} label={k} value={v} depth={depth + 1} />
        ))
      )}

      {open && isArr && (
        (value as unknown[]).map((item, i) => (
          <TreeNode key={i} label={`[${i}]`} value={item} depth={depth + 1} />
        ))
      )}
    </div>
  )
}

export default function WorldState() {
  const [world, setWorld] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    const load = () => api.getWorld().then(setWorld).catch(() => {})
    load()
    const interval = setInterval(load, 3000)
    return () => clearInterval(interval)
  }, [])

  if (!world) return (
    <div className="flex items-center justify-center h-full">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  return (
    <div className="page-enter space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>World State</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
            Live view of MERLIN's deterministic world projection
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
          <RefreshCw size={12} className="animate-spin" style={{ animationDuration: '3s' }} />
          Auto-refreshing
        </div>
      </div>

      <div className="glass-card p-4">
        <div className="flex items-center gap-2 mb-4">
          <Globe size={16} style={{ color: 'var(--color-accent)' }} />
          <span className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>WorldState</span>
        </div>

        <div className="font-mono overflow-auto max-h-[65vh]">
          {Object.entries(world).map(([key, value]) => (
            <TreeNode key={key} label={key} value={value} />
          ))}
        </div>
      </div>
    </div>
  )
}
