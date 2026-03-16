import { useEffect, useState } from 'react'
import { Cpu, HardDrive, MemoryStick, Clock, Zap, Battery, Activity, Wifi, Play, Pause, RotateCcw, Terminal, Rocket, Shield, TrendingUp } from 'lucide-react'
import { api, type SystemState } from '../lib/api'

/* ─── Circular Gauge with Glow ─── */
function Gauge({ value, label, color, glowColor, icon: Icon }: {
  value: number; label: string; color: string; glowColor: string; icon: typeof Cpu
}) {
  const radius = 45
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference
  const displayValue = Math.round(value)

  return (
    <div className="glass-card p-6 flex flex-col items-center gap-4 transition-transform duration-300 hover:scale-[1.02]">
      <div className="relative w-28 h-28">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={radius} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="6" />
          <defs>
            <linearGradient id={`gauge-grad-${label}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={color} />
              <stop offset="100%" stopColor={color} stopOpacity="0.4" />
            </linearGradient>
          </defs>
          <circle cx="50" cy="50" r={radius} fill="none"
            stroke={`url(#gauge-grad-${label})`} strokeWidth="6"
            strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
            className="gauge-ring transition-all duration-1000"
            style={{
              '--gauge-color': glowColor,
              filter: `drop-shadow(0 0 12px ${glowColor})`,
            } as React.CSSProperties} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center mb-1"
            style={{ background: color + '20' }}>
            <Icon size={18} style={{ color }} />
          </div>
          <span className="text-xl font-bold" style={{ color: 'var(--color-text-primary)' }}>
            {displayValue}%
          </span>
        </div>
      </div>
      <span className="text-xs font-medium tracking-wide uppercase" style={{ color: 'var(--color-text-secondary)' }}>
        {label}
      </span>
    </div>
  )
}

/* ─── Uptime Card with Spinning Ring ─── */
function UptimeCard({ uptime }: { uptime: number }) {
  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    if (h > 0) return `${h}h ${m}m`
    if (m > 0) return `${m}m ${s}s`
    return `${s}s`
  }

  return (
    <div className="glass-card p-6 flex flex-col items-center gap-4 transition-transform duration-300 hover:scale-[1.02]">
      <div className="relative w-28 h-28">
        <div className="absolute inset-0 rounded-full" style={{ border: '2px solid rgba(0,212,255,0.15)' }} />
        <div className="absolute inset-2 rounded-full animate-spin-slow"
          style={{ border: '1px dashed rgba(0,212,255,0.12)' }} />
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center mb-1"
            style={{ background: 'rgba(0,212,255,0.12)' }}>
            <Clock size={18} style={{ color: 'var(--color-accent)' }} />
          </div>
          <span className="text-xl font-bold font-mono" style={{ color: 'var(--color-text-primary)' }}>
            {formatUptime(uptime)}
          </span>
        </div>
      </div>
      <span className="text-xs font-medium tracking-wide uppercase" style={{ color: 'var(--color-text-secondary)' }}>
        Uptime
      </span>
    </div>
  )
}

/* ─── Battery Card with Shimmer ─── */
function BatteryCard({ percent, charging }: { percent: number; charging: boolean }) {
  return (
    <div className="glass-card p-5">
      <div className="flex items-center gap-3 mb-4">
        <div className="relative">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{ background: 'rgba(16,185,129,0.12)' }}>
            <Battery size={22} style={{
              color: percent < 20 ? 'var(--color-error)' : 'var(--color-success)'
            }} />
          </div>
          {charging && (
            <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center"
              style={{ background: 'var(--color-warning)', border: '2px solid var(--color-bg-primary)' }}>
              <Zap size={10} style={{ color: '#000', fill: '#000' }} />
            </div>
          )}
        </div>
        <div className="flex-1">
          <div className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>Battery</div>
          <div className="flex items-center gap-1.5">
            {charging && <Zap size={11} className="animate-pulse" style={{ color: 'var(--color-warning)' }} />}
            <span className="text-xs" style={{ color: charging ? 'var(--color-warning)' : 'var(--color-text-muted)' }}>
              {charging ? 'Charging' : 'On battery'}
            </span>
          </div>
        </div>
        <span className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>
          {Math.round(percent)}%
        </span>
      </div>
      {/* Progress bar with shimmer */}
      <div className="relative h-2.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.04)' }}>
        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-700" style={{
          width: `${percent}%`,
          background: percent < 20
            ? 'linear-gradient(90deg, #ef4444, #dc2626)'
            : percent < 50
            ? 'linear-gradient(90deg, #f59e0b, #d97706)'
            : 'linear-gradient(90deg, #10b981, #059669)',
        }} />
        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
          style={{
            width: `${percent}%`,
            background: 'linear-gradient(90deg, transparent 25%, rgba(255,255,255,0.2) 50%, transparent 75%)',
            backgroundSize: '200% 100%',
            animation: 'shimmer 2s linear infinite',
          }} />
        <div className="absolute inset-y-0 left-0 rounded-full glow-emerald"
          style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}

/* ─── API Server Card with Real Latency Graph ─── */
function ApiServerCard({ missionState }: { missionState: string }) {
  const [pingValues, setPingValues] = useState<number[]>([])
  const [online, setOnline] = useState(true)

  useEffect(() => {
    const measurePing = async () => {
      const start = performance.now()
      try {
        await api.getHealth()
        const latency = Math.round(performance.now() - start)
        setPingValues(prev => [...prev.slice(-19), latency])
        setOnline(true)
      } catch {
        setPingValues(prev => [...prev.slice(-19), 0])
        setOnline(false)
      }
    }
    measurePing()
    const interval = setInterval(measurePing, 1500)
    return () => clearInterval(interval)
  }, [])

  const maxPing = Math.max(...pingValues, 1)
  const latestPing = pingValues[pingValues.length - 1] || 0

  return (
    <div className="glass-card p-5">
      <div className="flex items-start gap-3 mb-4">
        <div className="relative">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{ background: 'rgba(0,212,255,0.12)' }}>
            <Wifi size={22} style={{ color: 'var(--color-accent)' }} />
          </div>
          <div className="absolute -bottom-1 -right-1 w-4 h-4">
            <span className="absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping"
              style={{ background: online ? 'var(--color-success)' : 'var(--color-error)' }} />
            <span className="relative inline-flex rounded-full h-4 w-4"
              style={{ background: online ? 'var(--color-success)' : 'var(--color-error)', border: '2px solid var(--color-bg-primary)' }} />
          </div>
        </div>
        <div className="flex-1">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>API Server</div>
              <div className="text-xs font-mono" style={{ color: 'var(--color-text-muted)' }}>localhost:8420</div>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
              style={{ background: online ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)', border: `1px solid ${online ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)'}` }}>
              <div className="w-1.5 h-1.5 rounded-full" style={{ background: online ? 'var(--color-success)' : 'var(--color-error)' }} />
              <span className="text-xs font-medium" style={{ color: online ? 'var(--color-success)' : 'var(--color-error)' }}>{online ? 'Online' : 'Offline'}</span>
            </div>
          </div>
        </div>
      </div>
      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Status</p>
          <p className="text-xs font-semibold" style={{ color: online ? 'var(--color-success)' : 'var(--color-error)' }}>{online ? 'Online' : 'Offline'}</p>
        </div>
        <div className="space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Mode</p>
          <p className="text-xs font-semibold" style={{ color: 'var(--color-text-primary)' }}>{missionState === 'idle' ? 'Standby' : 'Active'}</p>
        </div>
        <div className="space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Latency</p>
          <p className="text-xs font-semibold" style={{ color: 'var(--color-accent)' }}>{latestPing}ms</p>
        </div>
      </div>
      {/* Mini ping graph — real latency */}
      <div className="h-10 flex items-end gap-0.5">
        {pingValues.map((ping, i) => (
          <div
            key={i}
            className="flex-1 rounded-t transition-all duration-150"
            style={{
              height: `${Math.max((ping / maxPing) * 100, 4)}%`,
              background: ping === 0 ? 'rgba(239,68,68,0.3)' : 'rgba(0,212,255,0.25)',
            }}
            onMouseEnter={e => (e.currentTarget.style.background = ping === 0 ? 'rgba(239,68,68,0.5)' : 'rgba(0,212,255,0.5)')}
            onMouseLeave={e => (e.currentTarget.style.background = ping === 0 ? 'rgba(239,68,68,0.3)' : 'rgba(0,212,255,0.25)')}
          />
        ))}
      </div>
    </div>
  )
}

/* ─── Activity Chart — Real CPU data ─── */
function ActivityChart({ cpuPercent }: { cpuPercent: number }) {
  const [data, setData] = useState<number[]>([])

  // Accumulate real CPU readings over time
  useEffect(() => {
    if (cpuPercent === undefined) return
    setData(prev => {
      const updated = [...prev, Math.round(cpuPercent)]
      return updated.slice(-30) // keep last 30 readings
    })
  }, [cpuPercent])

  const maxValue = 100

  // Compute real trend: compare recent half vs older half average
  const computeTrend = () => {
    if (data.length < 6) return null
    const mid = Math.floor(data.length / 2)
    const older = data.slice(0, mid)
    const newer = data.slice(mid)
    const avgOlder = older.reduce((s, v) => s + v, 0) / older.length
    const avgNewer = newer.reduce((s, v) => s + v, 0) / newer.length
    if (avgOlder === 0) return null
    return ((avgNewer - avgOlder) / avgOlder) * 100
  }
  const trend = computeTrend()
  const trendUp = trend !== null && trend >= 0

  return (
    <div className="glass-card p-6 h-full">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: 'rgba(0,212,255,0.12)' }}>
            <Activity size={17} style={{ color: 'var(--color-accent)' }} />
          </div>
          <div>
            <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>CPU Activity</h3>
            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Live · {data.length} readings</p>
          </div>
        </div>
        {trend !== null && (
          <div className="flex items-center gap-1.5" style={{ color: trendUp ? 'var(--color-warning)' : 'var(--color-success)' }}>
            <TrendingUp size={14} style={{ transform: trendUp ? 'none' : 'scaleY(-1)' }} />
            <span className="text-xs font-medium">{trendUp ? '+' : ''}{trend.toFixed(1)}%</span>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="relative h-44">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-6 w-8 flex flex-col justify-between text-[10px]"
          style={{ color: 'var(--color-text-muted)' }}>
          <span>100</span>
          <span>50</span>
          <span>0</span>
        </div>

        {/* Grid lines */}
        <div className="absolute left-10 right-0 top-0 bottom-6">
          {[0, 1, 2].map(i => (
            <div key={i} className="absolute w-full"
              style={{ top: `${i * 50}%`, borderTop: '1px dashed rgba(255,255,255,0.04)' }} />
          ))}
        </div>

        {/* Bars */}
        <div className="absolute left-10 right-0 top-0 bottom-6 flex items-end gap-0.5">
          {data.map((value, i) => (
            <div
              key={i}
              className="flex-1 rounded-t transition-all duration-500 relative group"
              style={{
                height: `${Math.max((value / maxValue) * 100, 2)}%`,
                background: value > 80
                  ? 'linear-gradient(to top, rgba(239,68,68,0.7), rgba(239,68,68,0.3))'
                  : value > 50
                  ? 'linear-gradient(to top, rgba(245,158,11,0.7), rgba(245,158,11,0.3))'
                  : 'linear-gradient(to top, rgba(0,212,255,0.6), rgba(0,212,255,0.25))',
              }}
              onMouseEnter={e => (e.currentTarget.style.opacity = '0.8')}
              onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
            >
              <div className="absolute -top-7 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded text-[10px] font-mono opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10"
                style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text-primary)' }}>
                {value}%
              </div>
            </div>
          ))}
        </div>

        {/* X-axis labels */}
        <div className="absolute left-10 right-0 bottom-0 flex justify-between text-[10px]"
          style={{ color: 'var(--color-text-muted)' }}>
          <span>{data.length >= 30 ? '~90s ago' : 'Start'}</span>
          <span>Now</span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-5 mt-3 pt-3" style={{ borderTop: '1px solid var(--color-border)' }}>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--color-accent)' }} />
          <span className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>Normal (&lt;50%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#f59e0b' }} />
          <span className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>Moderate (50-80%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#ef4444' }} />
          <span className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>High (&gt;80%)</span>
        </div>
      </div>
    </div>
  )
}

/* ─── Quick Actions ─── */
const actions = [
  { icon: Play, label: 'Start', color: '#10b981', description: 'Launch services' },
  { icon: Pause, label: 'Pause', color: '#f59e0b', description: 'Pause all tasks' },
  { icon: RotateCcw, label: 'Restart', color: '#00d4ff', description: 'Restart system' },
  { icon: Terminal, label: 'Console', color: '#a78bfa', description: 'Open terminal' },
  { icon: Rocket, label: 'Deploy', color: '#00d4ff', description: 'Deploy changes' },
  { icon: Shield, label: 'Secure', color: '#10b981', description: 'Security scan' },
]

function QuickActions() {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl flex items-center justify-center"
          style={{ background: 'rgba(167,139,250,0.12)' }}>
          <Rocket size={17} style={{ color: '#a78bfa' }} />
        </div>
        <div>
          <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>Quick Actions</h3>
          <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Common operations</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        {actions.map(action => (
          <button
            key={action.label}
            className="group flex flex-col items-center gap-2 p-3.5 rounded-xl transition-all duration-300 hover:scale-105 active:scale-95"
            style={{
              background: `${action.color}10`,
              border: `1px solid ${action.color}20`,
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = `${action.color}20`
              e.currentTarget.style.borderColor = `${action.color}40`
              e.currentTarget.style.boxShadow = `0 0 20px ${action.color}30, 0 0 40px ${action.color}10`
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = `${action.color}10`
              e.currentTarget.style.borderColor = `${action.color}20`
              e.currentTarget.style.boxShadow = 'none'
            }}
          >
            <action.icon size={22} className="transition-transform group-hover:scale-110" style={{ color: action.color }} />
            <span className="text-xs font-medium" style={{ color: 'var(--color-text-primary)' }}>{action.label}</span>
            <span className="text-[10px] hidden sm:block" style={{ color: 'var(--color-text-muted)' }}>{action.description}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/* ─── Overview Page ─── */
export default function Overview() {
  const [system, setSystem] = useState<SystemState | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const load = () => {
      api.getSystem().then(setSystem).catch(e => setError(e.message))
    }
    load()
    const interval = setInterval(load, 3000)
    return () => clearInterval(interval)
  }, [])

  if (error) return (
    <div className="flex items-center justify-center h-[80vh]">
      <div className="glass-card p-10 text-center max-w-sm">
        <div className="w-14 h-14 rounded-2xl mx-auto mb-5 flex items-center justify-center"
          style={{ background: 'rgba(239,68,68,0.1)' }}>
          <Zap size={28} style={{ color: 'var(--color-error)' }} />
        </div>
        <p className="text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>Connection Failed</p>
        <p className="text-xs leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>{error}</p>
        <p className="text-xs mt-3" style={{ color: 'var(--color-text-muted)' }}>
          Make sure MERLIN is running with <code className="font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--color-bg-tertiary)' }}>--ui</code> flag
        </p>
      </div>
    </div>
  )

  if (!system) return (
    <div className="flex items-center justify-center h-[80vh]">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Gauge value={system.cpu_percent} label="CPU" color="#00d4ff" glowColor="rgba(0,212,255,0.4)" icon={Cpu} />
        <Gauge value={system.memory_percent} label="Memory" color="#a78bfa" glowColor="rgba(167,139,250,0.4)" icon={MemoryStick} />
        <Gauge value={system.disk_percent} label="Disk" color="#f59e0b" glowColor="rgba(245,158,11,0.4)" icon={HardDrive} />
        <UptimeCard uptime={system.uptime_seconds} />
      </div>

      {/* Battery + API Server */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {system.battery_percent !== undefined && (
          <BatteryCard percent={system.battery_percent} charging={!!system.battery_charging} />
        )}
        <ApiServerCard missionState={system.mission_state} />
      </div>

      {/* Activity Chart + Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <ActivityChart cpuPercent={system.cpu_percent} />
        </div>
        <QuickActions />
      </div>
    </div>
  )
}
