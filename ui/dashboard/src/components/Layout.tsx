import { Outlet, NavLink } from 'react-router-dom'
import {
  LayoutDashboard, MessageSquare, Clock, Brain,
  ScrollText, Settings, GitBranch, Globe, Mail,
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
    <div className="flex h-screen relative" style={{ zIndex: 1 }}>
      {/* Sidebar */}
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
          <div className="w-9 h-9 rounded-xl flex items-center justify-center text-sm font-bold relative"
            style={{
              background: 'linear-gradient(135deg, var(--color-accent), #0099cc)',
              color: '#000',
              boxShadow: '0 0 20px rgba(0,212,255,0.25)',
            }}>
            M
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

        {/* Bottom status */}
        <div className="p-4 mx-3 mb-3 rounded-xl" style={{ background: 'var(--color-bg-hover)' }}>
          <div className="flex items-center gap-2.5">
            <div className={`status-dot ${connected ? 'online' : 'offline'}`} />
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
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto relative" style={{ background: 'var(--color-bg-primary)' }}>
        <div className="p-7 page-enter">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
