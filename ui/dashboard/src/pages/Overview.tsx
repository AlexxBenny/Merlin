import { useEffect, useState } from 'react'
import { Cpu, HardDrive, MemoryStick, Clock, Zap, Battery, Activity, Wifi } from 'lucide-react'
import { api, type SystemState } from '../lib/api'

function Gauge({ value, label, color, icon: Icon }: {
  value: number; label: string; color: string; icon: typeof Cpu
}) {
  const radius = 45
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference
  const displayValue = Math.round(value)

  return (
    <div className="glass-card p-6 flex flex-col items-center gap-4">
      <div className="relative w-28 h-28">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          {/* Background track */}
          <circle cx="50" cy="50" r={radius} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="5" />
          {/* Gradient definition */}
          <defs>
            <linearGradient id={`gauge-grad-${label}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={color} />
              <stop offset="100%" stopColor={color} stopOpacity="0.4" />
            </linearGradient>
          </defs>
          {/* Value ring */}
          <circle cx="50" cy="50" r={radius} fill="none"
            stroke={`url(#gauge-grad-${label})`} strokeWidth="5"
            strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
            className="gauge-ring transition-all duration-1000"
            style={{ '--gauge-color': color } as React.CSSProperties} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <Icon size={18} style={{ color, opacity: 0.8 }} />
          <span className="text-xl font-bold mt-1" style={{ color: 'var(--color-text-primary)' }}>
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

function StatCard({ icon: Icon, label, value, color }: {
  icon: typeof Clock; label: string; value: string; color: string
}) {
  return (
    <div className="glass-card p-6 flex flex-col items-center justify-center gap-3">
      <div className="w-10 h-10 rounded-xl flex items-center justify-center"
        style={{ background: color + '15' }}>
        <Icon size={20} style={{ color }} />
      </div>
      <span className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>
        {value}
      </span>
      <span className="text-xs font-medium tracking-wide uppercase" style={{ color: 'var(--color-text-secondary)' }}>
        {label}
      </span>
    </div>
  )
}

function formatUptime(seconds: number) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

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
    <div className="page-enter space-y-6">
      {/* Header */}
      <div className="section-header">
        <div>
          <h1 className="section-title">System Overview</h1>
          <p className="section-subtitle">MERLIN runtime status and metrics</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`badge ${system.mission_state === 'idle' ? 'badge-success' : 'badge-info'}`}>
            <Activity size={10} />
            {system.mission_state}
          </span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Gauge value={system.cpu_percent} label="CPU" color="#00d4ff" icon={Cpu} />
        <Gauge value={system.memory_percent} label="Memory" color="#a78bfa" icon={MemoryStick} />
        <Gauge value={system.disk_percent} label="Disk" color="#f59e0b" icon={HardDrive} />
        <StatCard icon={Clock} label="Uptime" value={formatUptime(system.uptime_seconds)} color="var(--color-accent)" />
      </div>

      {/* Battery + System Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Battery */}
        {system.battery_percent !== undefined && (
          <div className="glass-card p-5">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                style={{ background: 'rgba(16,185,129,0.1)' }}>
                <Battery size={18} style={{
                  color: system.battery_percent < 20 ? 'var(--color-error)' : 'var(--color-success)'
                }} />
              </div>
              <div className="flex-1">
                <div className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>Battery</div>
                <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  {system.battery_charging ? '⚡ Charging' : 'On battery'}
                </div>
              </div>
              <span className="text-xl font-bold" style={{ color: 'var(--color-text-primary)' }}>
                {Math.round(system.battery_percent)}%
              </span>
            </div>
            <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.04)' }}>
              <div className="h-full rounded-full transition-all duration-700" style={{
                width: `${system.battery_percent}%`,
                background: system.battery_percent < 20
                  ? 'linear-gradient(90deg, #ef4444, #dc2626)'
                  : system.battery_percent < 50
                  ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                  : 'linear-gradient(90deg, #10b981, #059669)',
              }} />
            </div>
          </div>
        )}

        {/* Connection Info */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: 'rgba(0,212,255,0.1)' }}>
              <Wifi size={18} style={{ color: 'var(--color-accent)' }} />
            </div>
            <div>
              <div className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>API Server</div>
              <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>localhost:8420</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.02)' }}>
              <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Status</div>
              <div className="text-sm font-medium mt-1" style={{ color: 'var(--color-success)' }}>Online</div>
            </div>
            <div className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.02)' }}>
              <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Mode</div>
              <div className="text-sm font-medium mt-1" style={{ color: 'var(--color-text-primary)' }}>
                {system.mission_state === 'idle' ? 'Standby' : 'Active'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
