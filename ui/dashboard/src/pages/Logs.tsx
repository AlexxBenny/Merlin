import { useEffect, useRef, useState } from 'react'
import { createWebSocket, api, type LogEntry } from '../lib/api'

const levelConfig: Record<string, { color: string; bg: string }> = {
  DEBUG: { color: 'var(--color-text-muted)', bg: 'rgba(255,255,255,0.03)' },
  INFO: { color: '#00d4ff', bg: 'rgba(0,212,255,0.04)' },
  WARNING: { color: '#f59e0b', bg: 'rgba(245,158,11,0.04)' },
  ERROR: { color: '#ef4444', bg: 'rgba(239,68,68,0.04)' },
  CRITICAL: { color: '#ef4444', bg: 'rgba(239,68,68,0.08)' },
}

const levels = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR']

export default function Logs() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    api.getLogs(300).then(setLogs).catch(() => {})
  }, [])

  useEffect(() => {
    wsRef.current = createWebSocket('/ws/logs', (data) => {
      setLogs(prev => {
        const updated = [...prev, data as LogEntry]
        return updated.slice(-500)
      })
    })
    return () => wsRef.current?.close()
  }, [])

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const filtered = logs.filter(l => {
    if (filter !== 'ALL' && l.level !== filter) return false
    if (search && !l.message.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  return (
    <div className="page-enter flex flex-col h-full" style={{ maxHeight: 'calc(100vh - 80px)' }}>
      <div className="section-header">
        <div>
          <h1 className="section-title">Logs</h1>
          <p className="section-subtitle">Real-time log stream • {logs.length} entries</p>
        </div>
        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <div className="relative">
            <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)}
              className="sr-only peer" />
            <div className="w-8 h-4 rounded-full transition-colors peer-checked:bg-[rgba(0,212,255,0.3)]"
              style={{ background: 'rgba(255,255,255,0.08)' }}>
            </div>
            <div className="absolute top-0.5 left-0.5 w-3 h-3 rounded-full transition-all peer-checked:translate-x-4"
              style={{ background: 'var(--color-accent)' }}>
            </div>
          </div>
          <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Auto-scroll</span>
        </label>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-4 items-center flex-wrap">
        {levels.map(l => (
          <button key={l} onClick={() => setFilter(l)}
            className="px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200"
            style={{
              background: filter === l ? 'var(--color-accent-glow-strong)' : 'rgba(255,255,255,0.03)',
              color: filter === l ? 'var(--color-accent)' : 'var(--color-text-muted)',
              border: `1px solid ${filter === l ? 'rgba(0,212,255,0.2)' : 'var(--color-border)'}`,
            }}>
            {l}
          </button>
        ))}
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search logs..."
          className="input-field ml-auto text-xs py-1.5 px-3"
          style={{ width: '220px', borderRadius: '10px', fontSize: '0.75rem' }} />
      </div>

      {/* Log entries */}
      <div ref={scrollRef} className="flex-1 overflow-auto glass-card-static p-3 font-mono text-xs rounded-xl"
        style={{ lineHeight: '2' }}>
        {filtered.map((log, i) => {
          const config = levelConfig[log.level] || levelConfig.DEBUG
          return (
            <div key={i} className="px-3 py-0.5 rounded-md transition-colors flex gap-4 items-start"
              style={{ borderLeft: `2px solid ${config.color}` }}
              onMouseEnter={e => (e.currentTarget.style.background = config.bg)}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
              <span className="shrink-0" style={{ color: 'var(--color-text-muted)', minWidth: '72px' }}>
                {new Date(log.timestamp * 1000).toLocaleTimeString()}
              </span>
              <span className="font-semibold shrink-0" style={{ color: config.color, minWidth: '55px' }}>
                {log.level}
              </span>
              <span className="shrink-0" style={{ color: 'rgba(0,212,255,0.5)', minWidth: '90px' }}>{log.module}</span>
              <span className="break-all" style={{ color: 'var(--color-text-secondary)' }}>{log.message}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
