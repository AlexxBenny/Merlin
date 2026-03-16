import { useEffect, useState } from 'react'
import { Pause, Play, Trash2, RefreshCw, Calendar } from 'lucide-react'
import { api, type Job } from '../lib/api'

const statusStyles: Record<string, { badge: string; dot: string }> = {
  pending: { badge: 'badge-info', dot: '#818cf8' },
  running: { badge: 'badge-warning', dot: '#f59e0b' },
  completed: { badge: 'badge-success', dot: '#10b981' },
  failed: { badge: 'badge-error', dot: '#ef4444' },
  cancelled: { badge: 'badge-muted', dot: 'var(--color-text-muted)' },
  paused: { badge: 'badge-muted', dot: 'var(--color-text-muted)' },
}

function formatTime(ts: number | null) {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString()
}

export default function Scheduler() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)

  const load = () => {
    api.getJobs().then(setJobs).catch(() => {}).finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
    const interval = setInterval(load, 3000)
    return () => clearInterval(interval)
  }, [])

  const handlePause = async (id: string) => { await api.pauseJob(id); load() }
  const handleResume = async (id: string) => { await api.resumeJob(id); load() }
  const handleCancel = async (id: string) => { await api.cancelJob(id); load() }

  return (
    <div className="page-enter space-y-6">
      <div className="section-header">
        <div>
          <h1 className="section-title">Scheduler</h1>
          <p className="section-subtitle">Manage persistent jobs and recurring tasks</p>
        </div>
        <button onClick={load} className="btn-ghost">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      <div className="glass-card-static overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.02)' }}>
              {['Query', 'Type', 'Status', 'Priority', 'Next Run', 'Attempts', 'Actions'].map(h => (
                <th key={h} className="text-left px-5 py-3.5 text-xs font-semibold tracking-wider uppercase"
                  style={{ color: 'var(--color-text-muted)', borderBottom: '1px solid var(--color-border)' }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} className="text-center py-16" style={{ color: 'var(--color-text-muted)' }}>
                <div className="flex items-center justify-center gap-3">
                  <div className="animate-pulse-glow w-2 h-2 rounded-full" style={{ background: 'var(--color-accent)' }} />
                  Loading...
                </div>
              </td></tr>
            ) : jobs.length === 0 ? (
              <tr><td colSpan={7} className="text-center py-16">
                <Calendar size={28} style={{ color: 'var(--color-text-muted)', margin: '0 auto 12px' }} />
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>No jobs scheduled</p>
              </td></tr>
            ) : jobs.map(job => {
              const style = statusStyles[job.status] || statusStyles.cancelled
              return (
                <tr key={job.id} className="transition-colors duration-200"
                  style={{ borderBottom: '1px solid var(--color-border)' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-bg-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                  <td className="px-5 py-3.5" style={{ color: 'var(--color-text-primary)' }}>
                    <div className="max-w-xs truncate font-medium">{job.query}</div>
                    {job.error && <div className="text-xs mt-1" style={{ color: 'var(--color-error)' }}>{job.error}</div>}
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="font-mono text-xs px-2 py-1 rounded-md" style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                      {job.type}
                    </span>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className={`badge ${style.badge}`}>
                      <span className="w-1.5 h-1.5 rounded-full" style={{ background: style.dot }} />
                      {job.status}
                    </span>
                  </td>
                  <td className="px-5 py-3.5" style={{ color: 'var(--color-text-secondary)' }}>{job.priority}</td>
                  <td className="px-5 py-3.5 font-mono text-xs" style={{ color: 'var(--color-text-muted)' }}>{formatTime(job.next_run)}</td>
                  <td className="px-5 py-3.5" style={{ color: 'var(--color-text-secondary)' }}>{job.attempts}/{job.max_retries}</td>
                  <td className="px-5 py-3.5">
                    <div className="flex gap-1">
                      {job.status === 'pending' && (
                        <button onClick={() => handlePause(job.id)} className="p-2 rounded-lg transition-colors hover:bg-[rgba(245,158,11,0.1)]" title="Pause">
                          <Pause size={14} style={{ color: 'var(--color-warning)' }} />
                        </button>
                      )}
                      {job.status === 'paused' && (
                        <button onClick={() => handleResume(job.id)} className="p-2 rounded-lg transition-colors hover:bg-[rgba(16,185,129,0.1)]" title="Resume">
                          <Play size={14} style={{ color: 'var(--color-success)' }} />
                        </button>
                      )}
                      {['pending', 'paused'].includes(job.status) && (
                        <button onClick={() => handleCancel(job.id)} className="p-2 rounded-lg transition-colors hover:bg-[rgba(239,68,68,0.1)]" title="Cancel">
                          <Trash2 size={14} style={{ color: 'var(--color-error)' }} />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
