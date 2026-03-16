import { useEffect, useState } from 'react'
import { Globe, ChevronDown, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'

function TreeNode({ label, value, depth = 0 }: { label: string; value: unknown; depth?: number }) {
  const [open, setOpen] = useState(depth < 2)
  const isObj = typeof value === 'object' && value !== null && !Array.isArray(value)
  const isArr = Array.isArray(value)
  const isExpandable = isObj || isArr

  const valueColor = value === null || value === undefined
    ? 'var(--color-text-muted)'
    : typeof value === 'boolean'
    ? (value ? '#10b981' : '#ef4444')
    : typeof value === 'number'
    ? '#fbbf24'
    : typeof value === 'string'
    ? '#a78bfa'
    : 'var(--color-text-secondary)'

  return (
    <div style={{ paddingLeft: depth > 0 ? 16 : 0 }}>
      <div
        className="flex items-center gap-1.5 py-1 px-2.5 rounded-lg transition-colors"
        style={{ cursor: isExpandable ? 'pointer' : 'default' }}
        onClick={() => isExpandable && setOpen(!open)}
        onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-bg-hover)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
      >
        {/* Expand icon */}
        {isExpandable ? (
          <span className="transition-transform duration-200" style={{ transform: open ? 'rotate(0deg)' : 'rotate(-90deg)' }}>
            <ChevronDown size={12} style={{ color: 'var(--color-text-muted)' }} />
          </span>
        ) : (
          <span className="w-3 flex justify-center">
            <span className="w-1 h-1 rounded-full" style={{ background: valueColor }} />
          </span>
        )}

        {/* Key */}
        <span className="text-xs font-medium" style={{ color: isExpandable ? 'var(--color-accent-dim)' : 'var(--color-accent-dim)' }}>
          {label}
        </span>

        {/* Value */}
        {!isExpandable && (
          <span className="text-xs ml-1.5 font-mono" style={{ color: valueColor }}>
            {value === null ? 'null' : value === undefined ? 'undefined' :
              typeof value === 'string' ? `"${value}"` : String(value)}
          </span>
        )}

        {/* Array count / Object key count */}
        {isArr && (
          <span className="text-[10px] ml-1 px-1.5 py-0.5 rounded" style={{
            background: 'rgba(255,255,255,0.04)', color: 'var(--color-text-muted)'
          }}>
            {(value as unknown[]).length}
          </span>
        )}
        {isObj && (
          <span className="text-[10px] ml-1 px-1.5 py-0.5 rounded" style={{
            background: 'rgba(255,255,255,0.04)', color: 'var(--color-text-muted)'
          }}>
            {Object.keys(value as object).length}
          </span>
        )}
      </div>

      {/* Children with left border connector */}
      {open && isExpandable && (
        <div className="relative ml-2" style={{ borderLeft: '1px solid var(--color-border)' }}>
          {isObj && Object.entries(value as Record<string, unknown>).map(([k, v]) => (
            <TreeNode key={k} label={k} value={v} depth={depth + 1} />
          ))}
          {isArr && (value as unknown[]).map((item, i) => (
            <TreeNode key={i} label={`[${i}]`} value={item} depth={depth + 1} />
          ))}
        </div>
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
    <div className="flex items-center justify-center h-[80vh]">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  return (
    <div className="page-enter space-y-6">
      <div className="section-header">
        <div>
          <h1 className="section-title">World State</h1>
          <p className="section-subtitle">Live view of MERLIN's deterministic world projection</p>
        </div>
        <div className="flex items-center gap-2.5 text-xs" style={{ color: 'var(--color-text-muted)' }}>
          <RefreshCw size={11} className="animate-spin" style={{ animationDuration: '3s' }} />
          Auto-refreshing
        </div>
      </div>

      <div className="glass-card p-5">
        <div className="flex items-center gap-3 mb-5 pb-4" style={{ borderBottom: '1px solid var(--color-border)' }}>
          <div className="w-8 h-8 rounded-xl flex items-center justify-center"
            style={{ background: 'rgba(0,212,255,0.1)' }}>
            <Globe size={15} style={{ color: 'var(--color-accent)' }} />
          </div>
          <div>
            <span className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>WorldState</span>
            <span className="text-xs ml-2" style={{ color: 'var(--color-text-muted)' }}>
              {Object.keys(world).length} top-level keys
            </span>
          </div>
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
