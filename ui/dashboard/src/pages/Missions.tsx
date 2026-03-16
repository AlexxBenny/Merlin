import { useEffect, useState } from 'react'
import { GitBranch, CheckCircle, XCircle, SkipForward, Clock, Activity } from 'lucide-react'
import { api, type Mission } from '../lib/api'

const nodeStatusConfig: Record<string, { icon: typeof CheckCircle; color: string; bg: string }> = {
  completed: { icon: CheckCircle, color: '#10b981', bg: 'rgba(16,185,129,0.08)' },
  no_op: { icon: CheckCircle, color: 'var(--color-text-muted)', bg: 'rgba(255,255,255,0.03)' },
  failed: { icon: XCircle, color: '#ef4444', bg: 'rgba(239,68,68,0.08)' },
  skipped: { icon: SkipForward, color: 'var(--color-text-muted)', bg: 'rgba(255,255,255,0.03)' },
  timed_out: { icon: Clock, color: '#f59e0b', bg: 'rgba(245,158,11,0.08)' },
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
      <div className="section-header">
        <div>
          <h1 className="section-title">Mission Inspector</h1>
          <p className="section-subtitle">Visualize mission plans and execution results</p>
        </div>
        <span className="badge badge-muted">
          <Activity size={10} />
          {missions.length} missions
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Mission list */}
        <div className="glass-card p-4 space-y-2 max-h-[72vh] overflow-auto">
          <div className="px-1 py-2">
            <span className="text-[10px] font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
              Recent Missions
            </span>
          </div>
          {missions.length === 0 ? (
            <div className="py-12 text-center">
              <GitBranch size={28} style={{ color: 'var(--color-text-muted)', margin: '0 auto 12px' }} />
              <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>No missions yet</p>
            </div>
          ) : (
            missions.slice().reverse().map(m => {
              const hasFailures = m.nodes_failed.length > 0
              const isSelected = selected?.mission_id === m.mission_id
              return (
                <button key={m.mission_id} onClick={() => setSelected(m)}
                  className="w-full text-left p-3.5 rounded-xl transition-all duration-200"
                  style={{
                    background: isSelected ? 'var(--color-accent-glow-strong)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${isSelected ? 'rgba(0,212,255,0.15)' : 'transparent'}`,
                  }}
                  onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = 'var(--color-bg-hover)' }}
                  onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}>
                  <div className="flex items-center gap-2.5 mb-1.5">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: hasFailures ? '#ef4444' : '#10b981' }} />
                    <span className="text-xs font-mono font-medium truncate" style={{ color: 'var(--color-text-primary)' }}>
                      {m.mission_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 ml-4">
                    <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                      {m.nodes_executed.length} exec • {m.nodes_skipped.length} skip
                      {hasFailures && <span style={{ color: '#ef4444' }}> • {m.nodes_failed.length} fail</span>}
                    </span>
                    <span className="text-[10px] ml-auto" style={{ color: 'var(--color-text-muted)' }}>
                      {new Date(m.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                </button>
              )
            })
          )}
        </div>

        {/* DAG Detail */}
        <div className="lg:col-span-2 glass-card p-6">
          {!selected ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <GitBranch size={32} style={{ color: 'var(--color-text-muted)', margin: '0 auto 12px' }} />
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Select a mission to inspect</p>
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                    <span className="font-mono">{selected.mission_id}</span>
                  </h3>
                  <p className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                    {selected.nodes_executed.length + selected.nodes_failed.length + selected.nodes_skipped.length} nodes total
                  </p>
                </div>
                <div className="flex gap-2">
                  {selected.active_domain && <span className="badge badge-info">{selected.active_domain}</span>}
                  {selected.recovery_attempted && <span className="badge badge-warning">Recovery</span>}
                </div>
              </div>

              {/* Node list */}
              <div className="space-y-2">
                {selected.plan?.nodes ? (
                  selected.plan.nodes.map((node, idx) => {
                    const status = selected.node_statuses?.[node.id] || 'unknown'
                    const config = nodeStatusConfig[status] || { icon: Clock, color: 'var(--color-text-muted)', bg: 'rgba(255,255,255,0.03)' }
                    const { icon: StatusIcon, color, bg } = config

                    return (
                      <div key={node.id} className="relative">
                        {/* Connecting line */}
                        {idx > 0 && (
                          <div className="absolute -top-2 left-5 w-px h-2" style={{ background: 'var(--color-border)' }} />
                        )}
                        <div className="flex items-start gap-3.5 p-3.5 rounded-xl transition-colors"
                          style={{ background: bg, borderLeft: `3px solid ${color}` }}
                          onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-bg-hover)')}
                          onMouseLeave={e => (e.currentTarget.style.background = bg)}>
                          <StatusIcon size={15} style={{ color, marginTop: '2px' }} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="text-sm font-medium font-mono" style={{ color: 'var(--color-text-primary)' }}>
                                {node.id}
                              </span>
                              <span className="text-[10px] px-2 py-0.5 rounded-md font-medium"
                                style={{ background: color + '15', color, border: `1px solid ${color}25` }}>
                                {status}
                              </span>
                            </div>
                            <div className="text-xs mt-1.5" style={{ color: 'rgba(0,212,255,0.5)' }}>
                              skill: {node.skill}
                            </div>
                            {node.depends_on.length > 0 && (
                              <div className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                                depends: {node.depends_on.join(', ')}
                              </div>
                            )}
                            {Object.keys(node.inputs).length > 0 && (
                              <div className="text-xs mt-1 font-mono" style={{ color: 'var(--color-text-muted)' }}>
                                {JSON.stringify(node.inputs)}
                              </div>
                            )}
                          </div>
                          <span className="text-[10px] font-medium px-2 py-0.5 rounded-md shrink-0"
                            style={{ background: 'rgba(255,255,255,0.04)', color: 'var(--color-text-muted)' }}>
                            {node.mode}
                          </span>
                        </div>
                      </div>
                    )
                  })
                ) : (
                  <div className="space-y-2">
                    {selected.nodes_executed.map(n => (
                      <div key={n} className="flex items-center gap-3 px-3.5 py-3 rounded-xl"
                        style={{ background: 'rgba(16,185,129,0.06)', borderLeft: '3px solid #10b981' }}>
                        <CheckCircle size={14} style={{ color: '#10b981' }} />
                        <span className="text-sm font-mono" style={{ color: 'var(--color-text-primary)' }}>{n}</span>
                        <span className="badge badge-success ml-auto">executed</span>
                      </div>
                    ))}
                    {selected.nodes_skipped.map(n => (
                      <div key={n} className="flex items-center gap-3 px-3.5 py-3 rounded-xl"
                        style={{ background: 'rgba(255,255,255,0.02)', borderLeft: '3px solid var(--color-text-muted)' }}>
                        <SkipForward size={14} style={{ color: 'var(--color-text-muted)' }} />
                        <span className="text-sm font-mono" style={{ color: 'var(--color-text-muted)' }}>{n}</span>
                        <span className="badge badge-muted ml-auto">skipped</span>
                      </div>
                    ))}
                    {selected.nodes_failed.map(n => (
                      <div key={n} className="flex items-center gap-3 px-3.5 py-3 rounded-xl"
                        style={{ background: 'rgba(239,68,68,0.06)', borderLeft: '3px solid #ef4444' }}>
                        <XCircle size={14} style={{ color: '#ef4444' }} />
                        <span className="text-sm font-mono" style={{ color: 'var(--color-text-primary)' }}>{n}</span>
                        <span className="badge badge-error ml-auto">failed</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Active entity */}
              {selected.active_entity && (
                <div className="flex items-center gap-3 p-3.5 rounded-xl"
                  style={{ background: 'rgba(0,212,255,0.04)', border: '1px solid rgba(0,212,255,0.1)' }}>
                  <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Active entity</span>
                  <span className="text-sm font-medium font-mono" style={{ color: 'var(--color-accent)' }}>{selected.active_entity}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
