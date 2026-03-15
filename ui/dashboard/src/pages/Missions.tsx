import { useEffect, useState } from 'react'
import { GitBranch, CheckCircle, XCircle, SkipForward, Clock } from 'lucide-react'
import { api, type Mission } from '../lib/api'

const nodeStatusIcon: Record<string, { icon: typeof CheckCircle; color: string }> = {
  completed: { icon: CheckCircle, color: 'var(--color-success)' },
  no_op: { icon: CheckCircle, color: 'var(--color-text-muted)' },
  failed: { icon: XCircle, color: 'var(--color-error)' },
  skipped: { icon: SkipForward, color: 'var(--color-text-muted)' },
  timed_out: { icon: Clock, color: 'var(--color-warning)' },
}

export default function Missions() {
  const [missions, setMissions] = useState<Mission[]>([])
  const [selected, setSelected] = useState<Mission | null>(null)

  useEffect(() => {
    api.getMissions().then(setMissions).catch(() => {})
    const interval = setInterval(() => {
      api.getMissions().then(setMissions).catch(() => {})
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="page-enter space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>Mission Inspector</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
          Visualize mission plans and execution results
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Mission list */}
        <div className="glass-card p-4 space-y-2 max-h-[70vh] overflow-auto">
          <h3 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--color-accent-dim)' }}>
            Recent Missions ({missions.length})
          </h3>
          {missions.length === 0 ? (
            <p className="text-xs italic py-8 text-center" style={{ color: 'var(--color-text-muted)' }}>No missions yet</p>
          ) : (
            missions.slice().reverse().map(m => {
              const hasFailures = m.nodes_failed.length > 0
              const isSelected = selected?.mission_id === m.mission_id
              return (
                <button key={m.mission_id} onClick={() => setSelected(m)}
                  className="w-full text-left p-3 rounded-lg transition-all duration-200"
                  style={{
                    background: isSelected ? 'var(--color-accent-glow)' : 'var(--color-bg-tertiary)',
                    border: `1px solid ${isSelected ? 'var(--color-accent-dim)' : 'transparent'}`,
                  }}>
                  <div className="flex items-center gap-2 mb-1">
                    <GitBranch size={12} style={{ color: hasFailures ? 'var(--color-error)' : 'var(--color-success)' }} />
                    <span className="text-xs font-mono truncate" style={{ color: 'var(--color-text-primary)' }}>
                      {m.mission_id}
                    </span>
                  </div>
                  <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    {m.nodes_executed.length} executed • {m.nodes_skipped.length} skipped
                    {hasFailures && <span style={{ color: 'var(--color-error)' }}> • {m.nodes_failed.length} failed</span>}
                  </div>
                  <div className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                    {new Date(m.timestamp * 1000).toLocaleTimeString()}
                  </div>
                </button>
              )
            })
          )}
        </div>

        {/* DAG Visualization */}
        <div className="lg:col-span-2 glass-card p-5">
          {!selected ? (
            <div className="flex items-center justify-center h-60">
              <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Select a mission to inspect</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                  Mission: <span className="font-mono">{selected.mission_id}</span>
                </h3>
                <div className="flex gap-2">
                  {selected.active_domain && (
                    <span className="badge badge-info">{selected.active_domain}</span>
                  )}
                  {selected.recovery_attempted && (
                    <span className="badge badge-warning">Recovery</span>
                  )}
                </div>
              </div>

              {/* Node list (DAG representation) */}
              <div className="space-y-2">
                {selected.plan?.nodes ? (
                  selected.plan.nodes.map((node) => {
                    const status = selected.node_statuses?.[node.id] || 'unknown'
                    const { icon: StatusIcon, color } = nodeStatusIcon[status] || { icon: Clock, color: 'var(--color-text-muted)' }

                    return (
                      <div key={node.id} className="flex items-start gap-3 p-3 rounded-lg"
                        style={{ background: 'var(--color-bg-tertiary)', borderLeft: `3px solid ${color}` }}>
                        <StatusIcon size={16} style={{ color, marginTop: '2px' }} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium font-mono" style={{ color: 'var(--color-text-primary)' }}>
                              {node.id}
                            </span>
                            <span className="text-xs px-2 py-0.5 rounded" style={{ background: color + '20', color }}>
                              {status}
                            </span>
                          </div>
                          <div className="text-xs mt-1" style={{ color: 'var(--color-accent-dim)' }}>
                            skill: {node.skill}
                          </div>
                          {node.depends_on.length > 0 && (
                            <div className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                              depends on: {node.depends_on.join(', ')}
                            </div>
                          )}
                          {Object.keys(node.inputs).length > 0 && (
                            <div className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                              inputs: {JSON.stringify(node.inputs)}
                            </div>
                          )}
                        </div>
                        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                          {node.mode}
                        </span>
                      </div>
                    )
                  })
                ) : (
                  /* Fallback: show nodes from outcome */
                  <div className="space-y-1">
                    {selected.nodes_executed.map(n => (
                      <div key={n} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'var(--color-bg-tertiary)' }}>
                        <CheckCircle size={14} style={{ color: 'var(--color-success)' }} />
                        <span className="text-sm font-mono" style={{ color: 'var(--color-text-primary)' }}>{n}</span>
                        <span className="badge badge-success">executed</span>
                      </div>
                    ))}
                    {selected.nodes_skipped.map(n => (
                      <div key={n} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'var(--color-bg-tertiary)' }}>
                        <SkipForward size={14} style={{ color: 'var(--color-text-muted)' }} />
                        <span className="text-sm font-mono" style={{ color: 'var(--color-text-muted)' }}>{n}</span>
                        <span className="badge badge-muted">skipped</span>
                      </div>
                    ))}
                    {selected.nodes_failed.map(n => (
                      <div key={n} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'var(--color-bg-tertiary)' }}>
                        <XCircle size={14} style={{ color: 'var(--color-error)' }} />
                        <span className="text-sm font-mono" style={{ color: 'var(--color-text-primary)' }}>{n}</span>
                        <span className="badge badge-error">failed</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Active entity */}
              {selected.active_entity && (
                <div className="glass-card p-3 mt-4">
                  <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Active entity: </span>
                  <span className="text-sm font-medium" style={{ color: 'var(--color-accent)' }}>{selected.active_entity}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
