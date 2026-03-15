import { useEffect, useState } from 'react'
import { Cpu, HardDrive, MemoryStick, Clock, Zap, Battery } from 'lucide-react'
import { api, type SystemState } from '../lib/api'

function Gauge({ value, label, color, icon: Icon }: {
  value: number; label: string; color: string; icon: typeof Cpu
}) {
  const radius = 40
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference

  return (
    <div className="glass-card p-5 flex flex-col items-center gap-3">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={radius} fill="none" stroke="var(--color-border)" strokeWidth="6" />
          <circle cx="50" cy="50" r={radius} fill="none" stroke={color} strokeWidth="6"
            strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
            className="gauge-ring transition-all duration-1000" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <Icon size={16} style={{ color }} />
          <span className="text-lg font-bold" style={{ color: 'var(--color-text-primary)' }}>
            {Math.round(value)}%
          </span>
        </div>
      </div>
      <span className="text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>{label}</span>
    </div>
  )
}

function formatUptime(seconds: number) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
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
    <div className="flex items-center justify-center h-full">
      <div className="glass-card p-8 text-center">
        <Zap size={40} style={{ color: 'var(--color-error)', margin: '0 auto 16px' }} />
        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>{error}</p>
        <p className="text-xs mt-2" style={{ color: 'var(--color-text-muted)' }}>
          Make sure MERLIN is running with --ui flag
        </p>
      </div>
    </div>
  )

  if (!system) return (
    <div className="flex items-center justify-center h-full">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  return (
    <div className="page-enter space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>System Overview</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
            MERLIN runtime status and system metrics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`badge ${system.mission_state === 'idle' ? 'badge-success' : 'badge-info'}`}>
            {system.mission_state}
          </span>
        </div>
      </div>

      {/* Gauges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Gauge value={system.cpu_percent} label="CPU" color="var(--color-accent)" icon={Cpu} />
        <Gauge value={system.memory_percent} label="Memory" color="#a78bfa" icon={MemoryStick} />
        <Gauge value={system.disk_percent} label="Disk" color="#f59e0b" icon={HardDrive} />
        <div className="glass-card p-5 flex flex-col items-center gap-3 justify-center">
          <Clock size={24} style={{ color: 'var(--color-accent-dim)' }} />
          <span className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>
            {formatUptime(system.uptime_seconds)}
          </span>
          <span className="text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>Uptime</span>
        </div>
      </div>

      {/* Battery (if available) */}
      {system.battery_percent !== undefined && (
        <div className="glass-card p-4 flex items-center gap-4">
          <Battery size={20} style={{
            color: system.battery_percent < 20 ? 'var(--color-error)' : 'var(--color-success)'
          }} />
          <div className="flex-1">
            <div className="flex justify-between text-sm mb-1">
              <span style={{ color: 'var(--color-text-secondary)' }}>Battery</span>
              <span style={{ color: 'var(--color-text-primary)' }}>{Math.round(system.battery_percent)}%</span>
            </div>
            <div className="h-2 rounded-full" style={{ background: 'var(--color-bg-tertiary)' }}>
              <div className="h-full rounded-full transition-all duration-500" style={{
                width: `${system.battery_percent}%`,
                background: system.battery_percent < 20 ? 'var(--color-error)' :
                  system.battery_percent < 50 ? 'var(--color-warning)' : 'var(--color-success)',
              }} />
            </div>
          </div>
          <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {system.battery_charging ? '⚡ Charging' : 'On battery'}
          </span>
        </div>
      )}
    </div>
  )
}
