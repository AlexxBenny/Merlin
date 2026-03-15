import { Outlet, NavLink } from 'react-router-dom'
import {
  LayoutDashboard, MessageSquare, Clock, Brain,
  ScrollText, Settings, GitBranch, Globe, Activity,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/scheduler', icon: Clock, label: 'Scheduler' },
  { to: '/memory', icon: Brain, label: 'Memory' },
  { to: '/logs', icon: ScrollText, label: 'Logs' },
  { to: '/config', icon: Settings, label: 'Config' },
  { to: '/missions', icon: GitBranch, label: 'Missions' },
  { to: '/world', icon: Globe, label: 'World State' },
]

export default function Layout() {
  const [connected, setConnected] = useState(false)

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

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-56 flex flex-col border-r shrink-0"
        style={{ background: 'var(--color-bg-secondary)', borderColor: 'var(--color-border)' }}>
        {/* Logo */}
        <div className="p-5 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
            style={{ background: 'var(--color-accent)', color: 'var(--color-bg-primary)' }}>
            M
          </div>
          <div>
            <div className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>MERLIN</div>
            <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Dashboard</div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-2 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
                  isActive
                    ? 'font-medium'
                    : 'hover:bg-[var(--color-bg-hover)]'
                }`
              }
              style={({ isActive }) => ({
                background: isActive ? 'var(--color-accent-glow)' : undefined,
                color: isActive ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                borderLeft: isActive ? '2px solid var(--color-accent)' : '2px solid transparent',
              })}
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Status */}
        <div className="p-4 border-t" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-2 text-xs">
            <Activity size={12} style={{ color: connected ? 'var(--color-success)' : 'var(--color-error)' }} />
            <span style={{ color: 'var(--color-text-muted)' }}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto p-6 page-enter" style={{ background: 'var(--color-bg-primary)' }}>
        <Outlet />
      </main>
    </div>
  )
}
