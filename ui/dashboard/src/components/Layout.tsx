import { Outlet, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, MessageSquare, Clock, Brain,
  ScrollText, Settings, GitBranch, Globe, Mail,
  Sparkles, Zap,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/mail', icon: Mail, label: 'Mail' },
  { to: '/scheduler', icon: Clock, label: 'Scheduler' },
  { to: '/memory', icon: Brain, label: 'Memory' },
  { to: '/logs', icon: ScrollText, label: 'Logs' },
  { to: '/config', icon: Settings, label: 'Config' },
  { to: '/missions', icon: GitBranch, label: 'Missions' },
  { to: '/world', icon: Globe, label: 'World State' },
]

const pageTitles: Record<string, { title: string; subtitle: string }> = {
  '/': { title: 'System Overview', subtitle: 'MERLIN runtime status and metrics' },
  '/chat': { title: 'Chat', subtitle: 'Talk to MERLIN' },
  '/mail': { title: 'Mail', subtitle: 'Email integration' },
  '/scheduler': { title: 'Scheduler', subtitle: 'Manage persistent jobs and recurring tasks' },
  '/memory': { title: 'Memory', subtitle: 'User knowledge store' },
  '/logs': { title: 'Logs', subtitle: 'Real-time log stream' },
  '/config': { title: 'Configuration', subtitle: 'Edit MERLIN runtime settings' },
  '/missions': { title: 'Mission Inspector', subtitle: 'Visualize mission plans and execution results' },
  '/world': { title: 'World State', subtitle: "MERLIN's deterministic world projection" },
}

export default function Layout() {
  const [connected, setConnected] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [missionState, setMissionState] = useState('idle')
  const location = useLocation()

  useEffect(() => {
    const check = () => {
      api.getHealth()
        .then(() => setConnected(true))
        .catch(() => setConnected(false))
    }
    check()
    const interval = setInterval(check, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    api.getSystem()
      .then(s => setMissionState(s.mission_state))
      .catch(() => {})
    const interval = setInterval(() => {
      api.getSystem()
        .then(s => setMissionState(s.mission_state))
        .catch(() => {})
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const pageInfo = pageTitles[location.pathname] || { title: 'MERLIN', subtitle: '' }

  return (
    <div className="flex h-screen relative" style={{ zIndex: 1 }}>
      {/* ─── Sidebar ─── */}
      <aside className="w-60 flex flex-col shrink-0 relative"
        style={{
          background: 'linear-gradient(180deg, rgba(12,12,20,0.97) 0%, rgba(8,8,14,0.99) 100%)',
          borderRight: '1px solid var(--color-border)',
        }}>

        {/* Ambient glow at top */}
        <div className="absolute top-0 left-0 right-0 h-32 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at 50% -20%, rgba(0,212,255,0.06) 0%, transparent 70%)',
          }} />

        {/* Logo */}
        <div className="p-5 pb-6 flex items-center gap-3 relative">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center glow-cyan"
              style={{
                background: 'linear-gradient(135deg, var(--color-accent), var(--color-accent-secondary))',
              }}>
              <Sparkles size={20} style={{ color: '#000' }} />
            </div>
            {/* Online pulse dot */}
            <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2"
              style={{
                background: connected ? 'var(--color-success)' : 'var(--color-error)',
                borderColor: 'rgba(8,8,14,0.99)',
              }}>
              {connected && (
                <div className="absolute inset-0 rounded-full animate-ping"
                  style={{ background: 'var(--color-success)', opacity: 0.4 }} />
              )}
            </div>
          </div>
          <div>
            <div className="text-sm font-bold tracking-wide" style={{ color: 'var(--color-text-primary)' }}>MERLIN</div>
            <div className="text-[10px] font-medium tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>Dashboard</div>
          </div>
        </div>

        {/* Divider */}
        <div className="mx-4 mb-2" style={{ height: '1px', background: 'linear-gradient(90deg, transparent, var(--color-border), transparent)' }} />

        {/* Nav label */}
        <div className="px-5 py-2">
          <span className="text-[10px] font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
            Navigation
          </span>
        </div>

        {/* Nav items */}
        <nav className="flex-1 px-3 space-y-0.5">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `sidebar-nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={17} strokeWidth={1.8} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Bottom status — glass card */}
        <div className="p-4 mx-3 mb-3">
          <div className="glass-panel p-3.5">
            <div className="flex items-center gap-2.5">
              <div className="relative">
                <div className={`status-dot ${connected ? 'online' : 'offline'}`} />
                {connected && (
                  <div className="absolute inset-0 status-dot online animate-ping" style={{ opacity: 0.4 }} />
                )}
              </div>
              <div>
                <div className="text-xs font-medium" style={{ color: connected ? 'var(--color-success)' : 'var(--color-error)' }}>
                  {connected ? 'Connected' : 'Disconnected'}
                </div>
                <div className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
                  MERLIN Core
                </div>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* ─── Main content ─── */}
      <main className="flex-1 flex flex-col overflow-hidden relative" style={{ background: 'var(--color-bg-primary)' }}>
        {/* Background ambient glow orbs */}
        <div className="fixed pointer-events-none overflow-hidden" style={{ inset: 0, left: '240px' }}>
          <div className="absolute w-[500px] h-[500px] rounded-full blur-[120px] animate-pulse-glow"
            style={{ top: '-100px', left: '20%', background: 'rgba(0,212,255,0.04)' }} />
          <div className="absolute w-[400px] h-[400px] rounded-full blur-[100px] animate-pulse-glow"
            style={{ bottom: '-50px', right: '15%', background: 'rgba(124,92,252,0.04)', animationDelay: '1s' }} />
          <div className="absolute w-[300px] h-[300px] rounded-full blur-[80px] animate-pulse-glow"
            style={{ top: '40%', right: '30%', background: 'rgba(245,158,11,0.03)', animationDelay: '2s' }} />
        </div>

        {/* Sticky Header */}
        <header className="sticky top-0 z-40 shrink-0"
          style={{
            background: 'rgba(6,6,11,0.8)',
            backdropFilter: 'blur(16px)',
            borderBottom: '1px solid var(--color-border)',
          }}>
          <div className="flex items-center justify-between px-7 py-4">
            {/* Title */}
            <div>
              <h1 className="text-xl font-bold tracking-tight" style={{ color: 'var(--color-text-primary)' }}>
                {pageInfo.title}
              </h1>
              <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
                {pageInfo.subtitle}
              </p>
            </div>

            {/* Right controls */}
            <div className="flex items-center gap-3">
              {/* Live clock */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg"
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid var(--color-border)',
                }}>
                <span className="text-xs font-mono" style={{ color: 'var(--color-text-muted)' }}>
                  {currentTime.toLocaleTimeString()}
                </span>
              </div>

              {/* Status badge */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl glass-panel">
                <div className="relative">
                  <Zap size={14} style={{ color: missionState === 'idle' ? 'var(--color-warning)' : 'var(--color-accent)' }} />
                  <div className="absolute inset-0 animate-pulse-glow" style={{ opacity: 0.5 }}>
                    <Zap size={14} style={{ color: missionState === 'idle' ? 'var(--color-warning)' : 'var(--color-accent)', opacity: 0.5 }} />
                  </div>
                </div>
                <span className="text-xs font-semibold uppercase"
                  style={{ color: missionState === 'idle' ? 'var(--color-warning)' : 'var(--color-accent)' }}>
                  {missionState}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <div className="flex-1 overflow-auto relative">
          <div className="p-7 page-enter">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  )
}
