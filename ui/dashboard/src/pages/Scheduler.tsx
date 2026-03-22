import { useEffect, useState } from 'react'
import { Pause, Play, Trash2, RefreshCw, Calendar } from 'lucide-react'
import { api, type Job } from '../lib/api'

const statusBadge: Record<string, { cls: string; dot: string }> = {
  pending:   { cls: 'badge-cyan',  dot: 'var(--cyan)' },
  running:   { cls: 'badge-amber', dot: 'var(--amber)' },
  completed: { cls: 'badge-green', dot: 'var(--emerald)' },
  failed:    { cls: 'badge-red',   dot: 'var(--rose)' },
  cancelled: { cls: 'badge-muted', dot: 'var(--text-3)' },
  paused:    { cls: 'badge-muted', dot: 'var(--text-3)' },
}

function fmtTime(ts: number | null) {
  return ts ? new Date(ts * 1000).toLocaleString() : '—'
}

export default function Scheduler() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)

  const load = () => { api.getJobs().then(setJobs).catch(() => {}).finally(() => setLoading(false)) }

  useEffect(() => { load(); const i = setInterval(load, 3000); return () => clearInterval(i) }, [])

  const pause  = async (id: string) => { await api.pauseJob(id);  load() }
  const resume = async (id: string) => { await api.resumeJob(id); load() }
  const cancel = async (id: string) => { await api.cancelJob(id); load() }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button className="btn" onClick={load}><RefreshCw size={13} /> Refresh</button>
      </div>

      <div className="card" style={{ overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              {['Query', 'Type', 'Status', 'Priority', 'Next Run', 'Attempts', 'Actions'].map(h => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-3)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                  <div className="animate-bop" style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--cyan)' }} /> Loading...
                </div>
              </td></tr>
            ) : jobs.length === 0 ? (
              <tr><td colSpan={7} style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-3)' }}>
                <Calendar size={24} style={{ margin: '0 auto 10px', display: 'block' }} />
                <span style={{ fontSize: 12 }}>No jobs scheduled</span>
              </td></tr>
            ) : jobs.map(job => {
              const s = statusBadge[job.status] || statusBadge.cancelled
              return (
                <tr key={job.id}>
                  <td>
                    <div style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 500, color: 'var(--text-1)' }}>{job.query}</div>
                    {job.error && <div style={{ fontSize: 11, color: 'var(--rose)', marginTop: 3 }}>{job.error}</div>}
                  </td>
                  <td><span style={{ padding: '3px 8px', borderRadius: 6, background: 'var(--bg3)', color: 'var(--text-3)', fontSize: 11 }}>{job.type}</span></td>
                  <td><span className={`badge ${s.cls}`}><span style={{ width: 5, height: 5, borderRadius: '50%', background: s.dot }} />{job.status}</span></td>
                  <td>{job.priority}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-3)' }}>{fmtTime(job.next_run)}</td>
                  <td>{job.attempts}/{job.max_retries}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 4 }}>
                      {job.status === 'pending' && <button className="btn" style={{ padding: '5px 8px' }} onClick={() => pause(job.id)} title="Pause"><Pause size={14} style={{ color: 'var(--amber)' }} /></button>}
                      {job.status === 'paused' && <button className="btn btn-green" style={{ padding: '5px 8px' }} onClick={() => resume(job.id)} title="Resume"><Play size={14} /></button>}
                      {['pending', 'paused'].includes(job.status) && <button className="btn btn-red" style={{ padding: '5px 8px' }} onClick={() => cancel(job.id)} title="Cancel"><Trash2 size={14} /></button>}
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
