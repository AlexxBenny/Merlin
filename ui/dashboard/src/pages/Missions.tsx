import { useEffect, useState } from 'react'
import { GitBranch, CheckCircle, XCircle, SkipForward, Clock, Activity } from 'lucide-react'
import { api, type Mission } from '../lib/api'

const NODE_CFG: Record<string, { icon: typeof CheckCircle; color: string }> = {
  completed: { icon: CheckCircle,  color: 'var(--emerald)' },
  no_op:     { icon: CheckCircle,  color: 'var(--text-3)' },
  failed:    { icon: XCircle,      color: 'var(--rose)' },
  skipped:   { icon: SkipForward,  color: 'var(--text-3)' },
  timed_out: { icon: Clock,        color: 'var(--amber)' },
}

export default function Missions() {
  const [missions, setMissions] = useState<Mission[]>([])
  const [selected, setSelected] = useState<Mission | null>(null)

  useEffect(() => {
    const load = () => api.getMissions().then(setMissions).catch(() => {})
    load(); const i = setInterval(load, 5000); return () => clearInterval(i)
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <span className="badge badge-muted"><Activity size={10} /> {missions.length} missions</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 14 }}>
        {/* Mission list */}
        <div className="card" style={{ padding: 10, maxHeight: '70vh', overflowY: 'auto' }}>
          <div style={{ padding: '6px 8px 10px', fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.10em', color: 'var(--text-3)' }}>Recent</div>

          {missions.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '36px 0' }}>
              <GitBranch size={28} style={{ color: 'var(--text-3)', margin: '0 auto 10px', display: 'block' }} />
              <span style={{ fontSize: 12, color: 'var(--text-3)' }}>No missions yet</span>
            </div>
          ) : (
            missions.slice().reverse().map(m => {
              const hasFail = m.nodes_failed.length > 0
              const isSel = selected?.mission_id === m.mission_id
              return (
                <button key={m.mission_id} onClick={() => setSelected(m)} style={{
                  width: '100%', textAlign: 'left', padding: '10px 10px', borderRadius: 9,
                  marginBottom: 4, border: `1px solid ${isSel ? 'var(--border-accent)' : 'transparent'}`,
                  background: isSel ? 'var(--cyan-glow)' : 'transparent', cursor: 'pointer',
                  transition: 'all 0.15s', fontFamily: 'var(--font-mono)',
                }}
                  onMouseEnter={e => { if (!isSel) e.currentTarget.style.background = 'rgba(255,255,255,0.03)' }}
                  onMouseLeave={e => { if (!isSel) e.currentTarget.style.background = 'transparent' }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
                    <span style={{ width: 5, height: 5, borderRadius: '50%', background: hasFail ? 'var(--rose)' : 'var(--emerald)' }} />
                    <span style={{ fontSize: 11, color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.mission_id}</span>
                  </div>
                  <div style={{ paddingLeft: 12, fontSize: 10, color: 'var(--text-3)' }}>
                    {m.nodes_executed.length} exec · {m.nodes_skipped.length} skip
                    {hasFail && <span style={{ color: 'var(--rose)' }}> · {m.nodes_failed.length} fail</span>}
                  </div>
                </button>
              )
            })
          )}
        </div>

        {/* Detail */}
        <div className="card" style={{ padding: 22 }}>
          {!selected ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 240, flexDirection: 'column', gap: 12 }}>
              <GitBranch size={28} style={{ color: 'var(--text-3)' }} />
              <span style={{ fontSize: 13, color: 'var(--text-3)' }}>Select a mission to inspect</span>
            </div>
          ) : (
            <div>
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-1)' }}>{selected.mission_id}</h3>
                <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>
                  {selected.nodes_executed.length + selected.nodes_failed.length + selected.nodes_skipped.length} nodes total
                </p>
              </div>

              {selected.plan?.nodes ? (
                selected.plan.nodes.map((node, idx) => {
                  const status = selected.node_statuses?.[node.id] || 'unknown'
                  const cfg = NODE_CFG[status] || { icon: Clock, color: 'var(--text-3)' }
                  const { icon: SIcon, color } = cfg
                  return (
                    <div key={node.id}>
                      {idx > 0 && <div style={{ width: 1, height: 10, background: 'var(--border)', marginLeft: 20 }} />}
                      <div className="mission-node" style={{ background: `color-mix(in srgb, ${color} 8%, transparent)`, borderLeft: `2.5px solid ${color}` }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg4)')}
                        onMouseLeave={e => (e.currentTarget.style.background = `color-mix(in srgb, ${color} 8%, transparent)`)}
                      >
                        <SIcon size={14} style={{ color, marginTop: 2, flexShrink: 0 }} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                            <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)' }}>{node.id}</span>
                            <span style={{ fontSize: 10, padding: '1px 8px', borderRadius: 6, background: `color-mix(in srgb, ${color} 15%, transparent)`, color, border: `1px solid color-mix(in srgb, ${color} 25%, transparent)` }}>{status}</span>
                          </div>
                          <div style={{ fontSize: 11, color: 'rgba(0,210,255,0.50)', marginTop: 4 }}>skill: {node.skill}</div>
                          {node.depends_on.length > 0 && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>depends: {node.depends_on.join(', ')}</div>}
                        </div>
                        <span style={{ fontSize: 10, fontWeight: 500, padding: '2px 8px', borderRadius: 6, background: 'rgba(255,255,255,0.04)', color: 'var(--text-3)', flexShrink: 0 }}>{node.mode}</span>
                      </div>
                    </div>
                  )
                })
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {selected.nodes_executed.map(n => (
                    <div key={n} className="mission-node" style={{ background: 'rgba(16,185,129,0.06)', borderLeft: '2.5px solid var(--emerald)' }}>
                      <CheckCircle size={14} style={{ color: 'var(--emerald)' }} />
                      <span style={{ fontSize: 13, color: 'var(--text-1)', flex: 1 }}>{n}</span>
                      <span className="badge badge-green">executed</span>
                    </div>
                  ))}
                  {selected.nodes_skipped.map(n => (
                    <div key={n} className="mission-node" style={{ background: 'rgba(255,255,255,0.02)', borderLeft: '2.5px solid var(--text-3)' }}>
                      <SkipForward size={14} style={{ color: 'var(--text-3)' }} />
                      <span style={{ fontSize: 13, color: 'var(--text-3)', flex: 1 }}>{n}</span>
                      <span className="badge badge-muted">skipped</span>
                    </div>
                  ))}
                  {selected.nodes_failed.map(n => (
                    <div key={n} className="mission-node" style={{ background: 'rgba(244,63,94,0.06)', borderLeft: '2.5px solid var(--rose)' }}>
                      <XCircle size={14} style={{ color: 'var(--rose)' }} />
                      <span style={{ fontSize: 13, color: 'var(--text-1)', flex: 1 }}>{n}</span>
                      <span className="badge badge-red">failed</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
