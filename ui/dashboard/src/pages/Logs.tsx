import { useEffect, useRef, useState } from 'react'
import { createWebSocket, api, type LogEntry } from '../lib/api'

const levelColors: Record<string, string> = {
  DEBUG: 'var(--color-text-muted)',
  INFO: 'var(--color-accent)',
  WARNING: 'var(--color-warning)',
  ERROR: 'var(--color-error)',
  CRITICAL: 'var(--color-error)',
}

const levels = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR']

export default function Logs() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Initial load
  useEffect(() => {
    api.getLogs(300).then(setLogs).catch(() => {})
  }, [])

  // WebSocket
  useEffect(() => {
    wsRef.current = createWebSocket('/ws/logs', (data) => {
      setLogs(prev => {
        const updated = [...prev, data as LogEntry]
        return updated.slice(-500)
      })
    })
    return () => wsRef.current?.close()
  }, [])

  // Auto-scroll
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
    <div className="page-enter flex flex-col h-full" style={{ maxHeight: 'calc(100vh - 48px)' }}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>Logs</h1>
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
            Real-time log stream • {logs.length} entries
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: 'var(--color-text-muted)' }}>
          <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)}
            className="rounded" />
          Auto-scroll
        </label>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {levels.map(l => (
          <button key={l} onClick={() => setFilter(l)}
            className="px-3 py-1 rounded-full text-xs font-medium transition-colors"
            style={{
              background: filter === l ? 'var(--color-accent-glow)' : 'var(--color-bg-tertiary)',
              color: filter === l ? 'var(--color-accent)' : 'var(--color-text-muted)',
              border: `1px solid ${filter === l ? 'var(--color-accent-dim)' : 'var(--color-border)'}`,
            }}>
            {l}
          </button>
        ))}
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search logs..."
          className="ml-auto px-3 py-1 rounded-lg text-xs outline-none"
          style={{ background: 'var(--color-bg-input)', color: 'var(--color-text-primary)', border: '1px solid var(--color-border)', width: '200px' }} />
      </div>

      {/* Log entries */}
      <div ref={scrollRef} className="flex-1 overflow-auto glass-card p-2 font-mono text-xs" style={{ lineHeight: '1.8' }}>
        {filtered.map((log, i) => (
          <div key={i} className="px-2 py-0.5 hover:bg-[var(--color-bg-hover)] rounded flex gap-3"
            style={{ borderLeft: `2px solid ${levelColors[log.level] || 'var(--color-text-muted)'}` }}>
            <span style={{ color: 'var(--color-text-muted)', minWidth: '60px' }}>
              {new Date(log.timestamp * 1000).toLocaleTimeString()}
            </span>
            <span className="font-medium" style={{ color: levelColors[log.level], minWidth: '55px' }}>
              {log.level}
            </span>
            <span style={{ color: 'var(--color-accent-dim)', minWidth: '80px' }}>{log.module}</span>
            <span style={{ color: 'var(--color-text-secondary)' }}>{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
