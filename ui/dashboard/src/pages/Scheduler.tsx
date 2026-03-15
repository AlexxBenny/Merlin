import { useEffect, useState } from 'react'
import { Pause, Play, Trash2, RefreshCw } from 'lucide-react'
import { api, type Job } from '../lib/api'

const statusColor: Record<string, string> = {
  pending: 'badge-info',
  running: 'badge-warning',
  completed: 'badge-success',
  failed: 'badge-error',
  cancelled: 'badge-muted',
  paused: 'badge-muted',
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

  const handlePause = async (id: string) => {
    await api.pauseJob(id)
    load()
  }

  const handleResume = async (id: string) => {
    await api.resumeJob(id)
    load()
  }

  const handleCancel = async (id: string) => {
    await api.cancelJob(id)
    load()
  }

  return (
    <div className="page-enter space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>Scheduler</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
            Manage persistent jobs and recurring tasks
          </p>
        </div>
        <button onClick={load} className="p-2 rounded-lg transition-colors hover:bg-[var(--color-bg-hover)]">
          <RefreshCw size={16} style={{ color: 'var(--color-text-secondary)' }} />
        </button>
      </div>

      <div className="glass-card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
              {['Query', 'Type', 'Status', 'Priority', 'Next Run', 'Attempts', 'Actions'].map(h => (
                <th key={h} className="text-left px-4 py-3 font-medium" style={{ color: 'var(--color-text-muted)' }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} className="text-center py-12" style={{ color: 'var(--color-text-muted)' }}>Loading...</td></tr>
            ) : jobs.length === 0 ? (
              <tr><td colSpan={7} className="text-center py-12" style={{ color: 'var(--color-text-muted)' }}>No jobs scheduled</td></tr>
            ) : jobs.map(job => (
              <tr key={job.id} className="transition-colors hover:bg-[var(--color-bg-hover)]"
                style={{ borderBottom: '1px solid var(--color-border)' }}>
                <td className="px-4 py-3" style={{ color: 'var(--color-text-primary)' }}>
                  <div className="max-w-xs truncate">{job.query}</div>
                  {job.error && <div className="text-xs mt-1" style={{ color: 'var(--color-error)' }}>{job.error}</div>}
                </td>
                <td className="px-4 py-3" style={{ color: 'var(--color-text-secondary)' }}>{job.type}</td>
                <td className="px-4 py-3"><span className={`badge ${statusColor[job.status] || 'badge-muted'}`}>{job.status}</span></td>
                <td className="px-4 py-3" style={{ color: 'var(--color-text-secondary)' }}>{job.priority}</td>
                <td className="px-4 py-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>{formatTime(job.next_run)}</td>
                <td className="px-4 py-3" style={{ color: 'var(--color-text-secondary)' }}>{job.attempts}/{job.max_retries}</td>
                <td className="px-4 py-3">
                  <div className="flex gap-1">
                    {job.status === 'pending' && (
                      <button onClick={() => handlePause(job.id)} className="p-1.5 rounded-md hover:bg-[var(--color-bg-hover)]" title="Pause">
                        <Pause size={14} style={{ color: 'var(--color-warning)' }} />
                      </button>
                    )}
                    {job.status === 'paused' && (
                      <button onClick={() => handleResume(job.id)} className="p-1.5 rounded-md hover:bg-[var(--color-bg-hover)]" title="Resume">
                        <Play size={14} style={{ color: 'var(--color-success)' }} />
                      </button>
                    )}
                    {['pending', 'paused'].includes(job.status) && (
                      <button onClick={() => handleCancel(job.id)} className="p-1.5 rounded-md hover:bg-[var(--color-bg-hover)]" title="Cancel">
                        <Trash2 size={14} style={{ color: 'var(--color-error)' }} />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
