import { useEffect, useState } from 'react'
import { Globe, ChevronDown, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'

/* Recursive tree node */
function TreeNode({ label, value, depth = 0 }: { label: string; value: unknown; depth?: number }) {
  const [open, setOpen] = useState(depth < 2)
  const isObj = typeof value === 'object' && value !== null && !Array.isArray(value)
  const isArr = Array.isArray(value)
  const isExpandable = isObj || isArr

  const vc = value === null || value === undefined ? 'var(--text-3)'
    : typeof value === 'boolean' ? (value ? 'var(--emerald)' : 'var(--rose)')
    : typeof value === 'number' ? 'var(--amber)'
    : typeof value === 'string' ? '#a78bfa'
    : 'var(--text-2)'

  return (
    <div style={{ paddingLeft: depth > 0 ? 16 : 0 }}>
      <div
        className={`tree-row ${isExpandable ? 'expandable' : ''}`}
        onClick={() => isExpandable && setOpen(!open)}
      >
        {isExpandable ? (
          <span style={{ transition: 'transform 0.15s', transform: open ? 'rotate(0deg)' : 'rotate(-90deg)', display: 'flex' }}>
            <ChevronDown size={11} style={{ color: 'var(--text-3)' }} />
          </span>
        ) : (
          <span style={{ width: 12, display: 'flex', justifyContent: 'center' }}>
            <span style={{ width: 4, height: 4, borderRadius: '50%', background: vc }} />
          </span>
        )}

        <span style={{ fontSize: 12, fontWeight: 500, color: 'rgba(0,210,255,0.55)' }}>{label}</span>

        {!isExpandable && (
          <span style={{ fontSize: 12, marginLeft: 8, color: vc }}>
            {value === null ? 'null' : value === undefined ? 'undefined'
              : typeof value === 'string' ? `"${value}"` : String(value)}
          </span>
        )}

        {isArr && <span style={{ fontSize: 10, marginLeft: 6, padding: '1px 6px', borderRadius: 4, background: 'rgba(255,255,255,0.04)', color: 'var(--text-3)' }}>{(value as unknown[]).length}</span>}
        {isObj && <span style={{ fontSize: 10, marginLeft: 6, padding: '1px 6px', borderRadius: 4, background: 'rgba(255,255,255,0.04)', color: 'var(--text-3)' }}>{Object.keys(value as object).length}</span>}
      </div>

      {open && isExpandable && (
        <div style={{ marginLeft: 7, borderLeft: '1px solid var(--border)' }}>
          {isObj && Object.entries(value as Record<string, unknown>).map(([k, v]) => <TreeNode key={k} label={k} value={v} depth={depth + 1} />)}
          {isArr && (value as unknown[]).map((item, i) => <TreeNode key={i} label={`[${i}]`} value={item} depth={depth + 1} />)}
        </div>
      )}
    </div>
  )
}

export default function WorldState() {
  const [world, setWorld] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    const load = () => api.getWorld().then(setWorld).catch(() => {})
    load(); const i = setInterval(load, 3000); return () => clearInterval(i)
  }, [])

  if (!world) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '70vh' }}>
      <div className="animate-bop" style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--cyan)' }} />
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 8 }}>
        <RefreshCw size={11} className="animate-spin-slow" style={{ color: 'var(--text-3)' }} />
        <span style={{ fontSize: 11, color: 'var(--text-3)' }}>Auto-refreshing</span>
      </div>

      <div className="card" style={{ padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, paddingBottom: 14, borderBottom: '1px solid var(--border)' }}>
          <div style={{ width: 32, height: 32, borderRadius: 9, background: 'var(--cyan-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Globe size={15} style={{ color: 'var(--cyan)' }} />
          </div>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>WorldState</span>
          <span style={{ fontSize: 11, color: 'var(--text-3)', marginLeft: 10 }}>{Object.keys(world).length} top-level keys</span>
        </div>

        <div style={{ maxHeight: '65vh', overflowY: 'auto' }}>
          {Object.entries(world).map(([key, value]) => (
            <TreeNode key={key} label={key} value={value} />
          ))}
        </div>
      </div>
    </div>
  )
}
