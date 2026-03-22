import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, MessageSquare, Brain, GitBranch,
  Calendar, ScrollText, Globe, Mail, Settings2,
} from 'lucide-react'

const NAV_ITEMS = [
  { path: '/',          label: 'Overview',    icon: LayoutDashboard },
  { path: '/chat',      label: 'Chat',        icon: MessageSquare },
  { path: '/memory',    label: 'Memory',      icon: Brain },
  { path: '/missions',  label: 'Missions',    icon: GitBranch },
  { path: '/scheduler', label: 'Scheduler',   icon: Calendar },
  { path: '/logs',      label: 'Logs',        icon: ScrollText },
  { path: '/world',     label: 'World State',  icon: Globe },
  { path: '/mail',      label: 'Mail',        icon: Mail },
  { path: '/config',    label: 'Config',      icon: Settings2 },
]

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  '/':          { title: 'Overview',          subtitle: 'System metrics and status' },
  '/chat':      { title: 'Chat',              subtitle: 'Talk to MERLIN' },
  '/memory':    { title: 'Memory',            subtitle: 'User knowledge store · 5 domains' },
  '/missions':  { title: 'Mission Inspector', subtitle: 'Visualize plan execution and results' },
  '/scheduler': { title: 'Scheduler',         subtitle: 'Persistent jobs and recurring tasks' },
  '/logs':      { title: 'Logs',              subtitle: 'Real-time log stream' },
  '/world':     { title: 'World State',       subtitle: "MERLIN's deterministic world projection" },
  '/mail':      { title: 'Mail',              subtitle: 'Email drafts and compose' },
  '/config':    { title: 'Configuration',     subtitle: 'MERLIN runtime settings' },
}

export default function Layout() {
  const location = useLocation()
  const meta = PAGE_META[location.pathname] || PAGE_META['/']

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* ── Sidebar ───────────────────────────────── */}
      <aside style={{
        width: 190,
        flexShrink: 0,
        background: 'var(--bg1)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {/* Logo */}
        <div style={{
          padding: '18px 16px 14px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          {/* Orb */}
          <div style={{
            width: 28,
            height: 28,
            borderRadius: 8,
            background: 'linear-gradient(135deg, rgba(0,210,255,0.20), rgba(0,150,200,0.10))',
            border: '1px solid rgba(0,210,255,0.30)',
            boxShadow: '0 0 10px rgba(0,210,255,0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'var(--font-display)',
            fontSize: 12,
            fontWeight: 700,
            color: 'var(--cyan)',
            flexShrink: 0,
          }}>
            M
          </div>
          <div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: 13,
              fontWeight: 700,
              color: 'var(--text-1)',
              letterSpacing: '0.04em',
            }}>
              MERLIN
            </div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 9,
              color: 'var(--text-3)',
              letterSpacing: '0.10em',
              marginTop: 1,
            }}>
              AI ASSISTANT
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav style={{ flex: 1, padding: '10px 8px', overflowY: 'auto' }}>
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: 9,
                padding: '8px 9px',
                borderRadius: 8,
                border: 'none',
                textDecoration: 'none',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
                fontSize: 11.5,
                fontWeight: isActive ? 500 : 400,
                transition: 'all 0.15s',
                marginBottom: 2,
                borderLeft: `2px solid ${isActive ? 'var(--cyan)' : 'transparent'}`,
                background: isActive ? 'rgba(0,210,255,0.08)' : 'transparent',
                color: isActive ? 'var(--cyan)' : 'var(--text-3)',
              })}
              onMouseEnter={e => {
                const el = e.currentTarget
                if (!el.classList.contains('active')) {
                  el.style.background = 'rgba(255,255,255,0.03)'
                  el.style.color = 'var(--text-2)'
                }
              }}
              onMouseLeave={e => {
                const el = e.currentTarget
                if (!el.classList.contains('active')) {
                  el.style.background = 'transparent'
                  el.style.color = 'var(--text-3)'
                }
              }}
            >
              <Icon size={14} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div style={{
          padding: '10px 14px',
          borderTop: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 7,
        }}>
          <div style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: 'var(--emerald)',
            flexShrink: 0,
          }} />
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
          }}>
            localhost:8420
          </span>
        </div>
      </aside>

      {/* ── Main Content ──────────────────────────── */}
      <main style={{
        flex: 1,
        overflowY: 'auto',
        background: 'var(--bg0)',
        padding: '26px 28px',
      }}>
        {/* Page Header */}
        <div className="page-header">
          <div>
            <h1>{meta.title}</h1>
            <p>{meta.subtitle}</p>
          </div>
        </div>

        {/* Page Content */}
        <div className="fade-up">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
