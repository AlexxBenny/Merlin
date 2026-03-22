import { useEffect, useState } from 'react'
import {
  Cpu, HardDrive, MemoryStick, Clock, Zap, Battery, Activity, Wifi,
  TrendingUp, Play, Pause, RotateCcw, Terminal, Rocket, Shield,
} from 'lucide-react'
import { api, type SystemState } from '../lib/api'

/* ═══ Gauge Card ═══ */
function Gauge({ value, label, color, glowColor, icon: Icon }: {
  value: number; label: string; color: string; glowColor: string; icon: typeof Cpu
}) {
  const r = 38, C = 2 * Math.PI * r
  const offset = C - (value / 100) * C
  const v = Math.round(value)

  return (
    <div className="card" style={{ padding: 18, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <div style={{ position: 'relative', width: 90, height: 90 }}>
        <svg width="90" height="90" viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
          <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="5" />
          <defs>
            <linearGradient id={`g-${label}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={color} />
              <stop offset="100%" stopColor={color} stopOpacity="0.4" />
            </linearGradient>
          </defs>
          <circle
            cx="50" cy="50" r={r} fill="none"
            stroke={`url(#g-${label})`} strokeWidth="5" strokeLinecap="round"
            strokeDasharray={C} strokeDashoffset={offset}
            style={{ filter: `drop-shadow(0 0 6px ${glowColor})`, transition: 'stroke-dashoffset 1s ease' }}
          />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 3 }}>
          <Icon size={15} style={{ color }} />
          <span className="font-display" style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-1)' }}>{v}%</span>
        </div>
      </div>
      <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.10em' }}>{label}</span>
    </div>
  )
}

/* ═══ Uptime Card ═══ */
function UptimeCard({ uptime }: { uptime: number }) {
  const h = Math.floor(uptime / 3600), m = Math.floor((uptime % 3600) / 60), s = Math.floor(uptime % 60)
  const fmt = h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`

  return (
    <div className="card" style={{ padding: 18, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <div style={{ position: 'relative', width: 90, height: 90 }}>
        <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '1.5px solid rgba(0,210,255,0.12)' }} />
        <div className="animate-spin-slow" style={{ position: 'absolute', inset: 8, borderRadius: '50%', border: '1px dashed rgba(0,210,255,0.08)' }} />
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 3 }}>
          <Clock size={15} style={{ color: 'var(--cyan)' }} />
          <span className="font-display" style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-1)' }}>{fmt}</span>
        </div>
      </div>
      <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.10em' }}>Uptime</span>
    </div>
  )
}

/* ═══ Battery Card ═══ */
function BatteryCard({ percent, charging }: { percent: number; charging: boolean }) {
  const barColor = percent < 20 ? 'linear-gradient(90deg,#f43f5e,#dc2626)'
    : percent < 50 ? 'linear-gradient(90deg,#f59e0b,#d97706)'
    : 'linear-gradient(90deg,#10b981,#059669)'

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ position: 'relative' }}>
          <div style={{ width: 38, height: 38, borderRadius: 10, background: 'rgba(16,185,129,0.10)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Battery size={22} style={{ color: percent < 20 ? 'var(--rose)' : 'var(--emerald)' }} />
          </div>
          {charging && (
            <div style={{ position: 'absolute', bottom: -2, right: -2, width: 16, height: 16, borderRadius: '50%', background: 'var(--amber)', border: '2px solid var(--bg1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Zap size={9} style={{ color: '#000', fill: '#000' }} />
            </div>
          )}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>Battery</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {charging && <Zap size={11} style={{ color: 'var(--amber)' }} />}
            <span style={{ fontSize: 11, color: charging ? 'var(--amber)' : 'var(--text-3)' }}>{charging ? 'Charging' : 'On battery'}</span>
          </div>
        </div>
        <span className="font-display" style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-1)' }}>{Math.round(percent)}%</span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: 'rgba(255,255,255,0.04)', overflow: 'hidden', marginTop: 10, position: 'relative' }}>
        <div style={{ position: 'absolute', inset: 0, width: `${percent}%`, borderRadius: 3, background: barColor, transition: 'width 0.7s' }} />
      </div>
    </div>
  )
}

/* ═══ API Server Card ═══ */
function ApiServerCard({ missionState }: { missionState: string }) {
  const [pings, setPings] = useState<number[]>([])
  const [online, setOnline] = useState(true)

  useEffect(() => {
    const measure = async () => {
      const t0 = performance.now()
      try { await api.getHealth(); setPings(p => [...p.slice(-19), Math.round(performance.now() - t0)]); setOnline(true) }
      catch { setPings(p => [...p.slice(-19), 0]); setOnline(false) }
    }
    measure()
    const i = setInterval(measure, 1500)
    return () => clearInterval(i)
  }, [])

  const maxP = Math.max(...pings, 1), latest = pings[pings.length - 1] || 0

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 14 }}>
        <div style={{ position: 'relative' }}>
          <div style={{ width: 38, height: 38, borderRadius: 10, background: 'var(--cyan-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Wifi size={22} style={{ color: 'var(--cyan)' }} />
          </div>
          <div style={{ position: 'absolute', bottom: -2, right: -2, width: 11, height: 11 }}>
            <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: online ? 'var(--emerald)' : 'var(--rose)', opacity: 0.75 }} className="animate-ping-dot" />
            <span style={{ position: 'relative', display: 'block', width: 11, height: 11, borderRadius: '50%', background: online ? 'var(--emerald)' : 'var(--rose)', border: '2px solid var(--bg1)' }} />
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>API Server</div>
              <div style={{ fontSize: 10, color: 'var(--text-3)' }}>localhost:8420</div>
            </div>
            <span className={`badge ${online ? 'badge-green' : 'badge-red'}`}>
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: online ? 'var(--emerald)' : 'var(--rose)' }} />
              {online ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginBottom: 14 }}>
        <div><p style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-3)' }}>Status</p><p style={{ fontSize: 12, fontWeight: 600, color: online ? 'var(--emerald)' : 'var(--rose)' }}>{online ? 'Online' : 'Offline'}</p></div>
        <div><p style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-3)' }}>Mode</p><p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-1)' }}>{missionState === 'idle' ? 'Standby' : 'Active'}</p></div>
        <div><p style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-3)' }}>Latency</p><p style={{ fontSize: 12, fontWeight: 600, color: 'var(--cyan)' }}>{latest}ms</p></div>
      </div>

      <div style={{ height: 28, display: 'flex', alignItems: 'flex-end', gap: 1 }}>
        {pings.map((p, i) => (
          <div key={i} style={{ flex: 1, borderRadius: '2px 2px 0 0', height: `${Math.max((p / maxP) * 100, 4)}%`, background: p === 0 ? 'rgba(244,63,94,0.30)' : 'rgba(0,210,255,0.30)', transition: 'height 0.15s' }} />
        ))}
      </div>
    </div>
  )
}

/* ═══ Activity Chart ═══ */
function ActivityChart({ cpuPercent }: { cpuPercent: number }) {
  const [data, setData] = useState<number[]>([])

  useEffect(() => {
    if (cpuPercent === undefined) return
    setData(p => [...p, Math.round(cpuPercent)].slice(-30))
  }, [cpuPercent])

  const trend = (() => {
    if (data.length < 6) return null
    const mid = Math.floor(data.length / 2)
    const a = data.slice(0, mid).reduce((s, v) => s + v, 0) / mid
    const b = data.slice(mid).reduce((s, v) => s + v, 0) / (data.length - mid)
    if (a === 0) return null
    return ((b - a) / a) * 100
  })()
  const trendUp = trend !== null && trend >= 0

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 9, background: 'var(--cyan-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Activity size={17} style={{ color: 'var(--cyan)' }} />
          </div>
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>CPU Activity</h3>
            <p style={{ fontSize: 11, color: 'var(--text-3)' }}>Live · {data.length} readings</p>
          </div>
        </div>
        {trend !== null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: trendUp ? 'var(--amber)' : 'var(--emerald)' }}>
            <TrendingUp size={14} style={{ transform: trendUp ? 'none' : 'scaleY(-1)' }} />
            <span style={{ fontSize: 12, fontWeight: 500 }}>{trendUp ? '+' : ''}{trend.toFixed(1)}%</span>
          </div>
        )}
      </div>

      <div style={{ position: 'relative', height: 100 }}>
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 18, width: 28, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-3)' }}>
          <span>100</span><span>50</span><span>0</span>
        </div>
        {[0, 1, 2].map(i => (
          <div key={i} style={{ position: 'absolute', left: 32, right: 0, top: `${i * 41}px`, borderTop: '1px dashed rgba(255,255,255,0.03)' }} />
        ))}
        <div style={{ position: 'absolute', left: 32, right: 0, top: 0, bottom: 18, display: 'flex', alignItems: 'flex-end', gap: 2 }}>
          {data.map((v, i) => (
            <div key={i} style={{
              flex: 1, borderRadius: '2px 2px 0 0', transition: 'height 0.5s',
              height: `${Math.max((v / 100) * 100, 2)}%`,
              background: v > 80 ? 'rgba(244,63,94,0.50)' : v > 50 ? 'rgba(245,158,11,0.50)' : 'rgba(0,210,255,0.40)',
            }} />
          ))}
        </div>
        <div style={{ position: 'absolute', left: 32, right: 0, bottom: 0, display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-3)' }}>
          <span>{data.length >= 30 ? '~60s ago' : 'Start'}</span><span>Now</span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border)' }}>
        {[{ c: 'var(--cyan)', l: 'Normal' }, { c: '#f59e0b', l: 'Moderate' }, { c: '#f43f5e', l: 'High' }].map(({ c, l }) => (
          <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: c }} />
            <span style={{ fontSize: 10, color: 'var(--text-3)' }}>{l}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ═══ Quick Actions ═══ */
const ACTIONS = [
  { icon: Play,       label: 'Start',   color: '#10b981' },
  { icon: Pause,      label: 'Pause',   color: '#f59e0b' },
  { icon: RotateCcw,  label: 'Restart', color: '#00d2ff' },
  { icon: Terminal,   label: 'Console', color: '#8b5cf6' },
  { icon: Rocket,     label: 'Deploy',  color: '#00d2ff' },
  { icon: Shield,     label: 'Secure',  color: '#10b981' },
]

function QuickActions() {
  return (
    <div className="card" style={{ padding: 18, width: 200 }}>
      <div style={{ fontSize: 11, textTransform: 'uppercase', color: 'var(--text-3)', letterSpacing: '0.10em', marginBottom: 10 }}>Quick Actions</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {ACTIONS.map(({ icon: Icon, label, color }) => (
          <button key={label} style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5,
            padding: '10px 6px', borderRadius: 9, border: `1px solid ${color}25`,
            background: `${color}0d`, color, cursor: 'pointer', transition: 'all 0.15s',
          }}
            onMouseEnter={e => { e.currentTarget.style.background = `${color}22`; e.currentTarget.style.transform = 'scale(1.04)' }}
            onMouseLeave={e => { e.currentTarget.style.background = `${color}0d`; e.currentTarget.style.transform = 'scale(1)' }}
          >
            <Icon size={15} />
            <span style={{ fontSize: 10, fontWeight: 500 }}>{label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/* ═══ Overview Page ═══ */
export default function Overview() {
  const [sys, setSys] = useState<SystemState | null>(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    const load = () => api.getSystem().then(setSys).catch(e => setErr(e.message))
    load()
    const i = setInterval(load, 3000)
    return () => clearInterval(i)
  }, [])

  if (err) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '70vh' }}>
      <div className="card" style={{ padding: 40, textAlign: 'center', maxWidth: 340 }}>
        <Zap size={28} style={{ color: 'var(--rose)', marginBottom: 12 }} />
        <p style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)', marginBottom: 6 }}>Connection Failed</p>
        <p style={{ fontSize: 12, color: 'var(--text-3)' }}>{err}</p>
        <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 10 }}>
          Run MERLIN with <code style={{ padding: '2px 6px', borderRadius: 4, background: 'var(--bg3)' }}>--ui</code> flag
        </p>
      </div>
    </div>
  )

  if (!sys) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '70vh' }}>
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--cyan)' }} className="animate-bop" />
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Row 1: Gauges */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <Gauge value={sys.cpu_percent} label="CPU" color="#00d2ff" glowColor="rgba(0,210,255,0.50)" icon={Cpu} />
        <Gauge value={sys.memory_percent} label="Memory" color="#8b5cf6" glowColor="rgba(139,92,246,0.50)" icon={MemoryStick} />
        <Gauge value={sys.disk_percent} label="Disk" color="#f59e0b" glowColor="rgba(245,158,11,0.50)" icon={HardDrive} />
        <UptimeCard uptime={sys.uptime_seconds} />
      </div>

      {/* Row 2: Battery + API */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {sys.battery_percent !== undefined && <BatteryCard percent={sys.battery_percent} charging={!!sys.battery_charging} />}
        <ApiServerCard missionState={sys.mission_state} />
      </div>

      {/* Row 3: Chart + Quick Actions */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 200px', gap: 12 }}>
        <ActivityChart cpuPercent={sys.cpu_percent} />
        <QuickActions />
      </div>
    </div>
  )
}
