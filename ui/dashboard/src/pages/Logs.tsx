import { useEffect, useRef, useState } from 'react'
import { createWebSocket, api, type LogEntry } from '../lib/api'

const LEVEL_CFG: Record<string, { color: string; hover: string }> = {
  DEBUG:    { color: 'var(--text-3)', hover: 'rgba(255,255,255,0.10)' },
  INFO:     { color: 'var(--cyan)',   hover: 'rgba(0,210,255,0.08)' },
  WARNING:  { color: 'var(--amber)',  hover: 'rgba(245,158,11,0.08)' },
  ERROR:    { color: 'var(--rose)',   hover: 'rgba(244,63,94,0.08)' },
  CRITICAL: { color: 'var(--rose)',   hover: 'rgba(244,63,94,0.15)' },
}

const LEVELS = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR']

export default function Logs() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => { api.getLogs(300).then(setLogs).catch(() => {}) }, [])

  useEffect(() => {
    wsRef.current = createWebSocket('/ws/logs', data => {
      setLogs(p => [...p, data as LogEntry].slice(-500))
    })
    return () => wsRef.current?.close()
  }, [])

  useEffect(() => {
    if (autoScroll && scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [logs, autoScroll])

  const filtered = logs.filter(l => {
    if (filter !== 'ALL' && l.level !== filter) return false
    if (search && !l.message.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 96px)', gap: 16 }}>
      {/* Toggle */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{logs.length} entries</span>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
          <div
            className={`toggle-track ${autoScroll ? 'active' : ''}`}
            onClick={() => setAutoScroll(!autoScroll)}
          >
            <div className="toggle-thumb" />
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-3)' }}>Auto-scroll</span>
        </label>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
        {LEVELS.map(l => (
          <button key={l} onClick={() => setFilter(l)} className="btn" style={{
            padding: '5px 12px', fontSize: 11,
            ...(filter === l ? { background: 'var(--cyan-dim)', color: 'var(--cyan)', borderColor: 'rgba(0,210,255,0.25)' } : {}),
          }}>{l}</button>
        ))}
        <input
          className="input"
          style={{ marginLeft: 'auto', width: 200, padding: '6px 12px', fontSize: 12 }}
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search logs…"
        />
      </div>

      {/* Log entries */}
      <div ref={scrollRef} className="card" style={{ flex: 1, overflowY: 'auto', padding: '8px 6px' }}>
        {filtered.map((log, i) => {
          const cfg = LEVEL_CFG[log.level] || LEVEL_CFG.DEBUG
          return (
            <div key={i} style={{
              display: 'flex', gap: 16, padding: '5px 10px', borderRadius: 6,
              borderLeft: `2px solid ${cfg.color}`, marginBottom: 2, transition: 'background 0.12s',
            }}
              onMouseEnter={e => (e.currentTarget.style.background = cfg.hover)}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            >
              <span style={{ flexShrink: 0, fontSize: 11, color: 'var(--text-3)', minWidth: 70 }}>
                {new Date(log.timestamp * 1000).toLocaleTimeString()}
              </span>
              <span style={{ flexShrink: 0, fontSize: 11, fontWeight: 600, color: cfg.color, minWidth: 58 }}>
                {log.level}
              </span>
              <span style={{ flexShrink: 0, fontSize: 11, color: 'rgba(0,210,255,0.45)', minWidth: 120 }}>
                {log.module}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-2)', wordBreak: 'break-all', flex: 1 }}>
                {log.message}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
