import { useEffect, useState } from 'react'
import { Save, Settings2 } from 'lucide-react'
import { api } from '../lib/api'

export default function Config() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null)
  const [meta, setMeta] = useState<Record<string, { label: string; description: string; type: string }>>({})
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    api.getConfig().then(data => {
      const fieldMeta = (data._field_metadata || {}) as typeof meta
      setMeta(fieldMeta)
      const { _field_metadata, ...rest } = data
      setConfig(rest)
    }).catch(() => {})
  }, [])

  const handleSave = async (section: string, key: string, value: unknown) => {
    setSaving(true)
    setMessage('')
    try {
      const res = await api.updateConfig({ [section]: { [key]: value } })
      setMessage(res.message || 'Saved')
    } catch (e: unknown) {
      setMessage(e instanceof Error ? e.message : 'Save failed')
    }
    setSaving(false)
    setTimeout(() => setMessage(''), 3000)
  }

  if (!config) return (
    <div className="flex items-center justify-center h-[80vh]">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  return (
    <div className="page-enter space-y-6">
      <div className="section-header">
        <div>
          <h1 className="section-title">Configuration</h1>
          <p className="section-subtitle">Edit MERLIN runtime settings</p>
        </div>
        {message && (
          <span className={`badge ${message.toLowerCase().includes('fail') ? 'badge-error' : 'badge-success'}`}>
            {message}
          </span>
        )}
      </div>

      {Object.entries(config).map(([section, values]) => {
        if (typeof values !== 'object' || values === null) return null
        const sectionData = values as Record<string, unknown>

        return (
          <div key={section} className="glass-card overflow-hidden">
            {/* Section header */}
            <div className="px-6 py-4 flex items-center gap-3"
              style={{ background: 'rgba(255,255,255,0.015)', borderBottom: '1px solid var(--color-border)' }}>
              <Settings2 size={14} style={{ color: 'var(--color-accent-dim)' }} />
              <h3 className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--color-accent-dim)' }}>
                {section}
              </h3>
            </div>

            <div className="p-6 space-y-1">
              {Object.entries(sectionData).map(([key, value]) => {
                if (typeof value === 'object' && value !== null) {
                  return Object.entries(value as Record<string, unknown>).map(([subKey, subVal]) => (
                    <ConfigField
                      key={`${section}.${key}.${subKey}`}
                      label={meta[`${section}.${key}.${subKey}`]?.label || `${key}.${subKey}`}
                      description={meta[`${section}.${key}.${subKey}`]?.description}
                      value={subVal}
                      onSave={(v) => handleSave(section, key, { [subKey]: v })}
                      saving={saving}
                    />
                  ))
                }
                return (
                  <ConfigField
                    key={`${section}.${key}`}
                    label={meta[`${section}.${key}`]?.label || key}
                    description={meta[`${section}.${key}`]?.description}
                    value={value}
                    onSave={(v) => handleSave(section, key, v)}
                    saving={saving}
                  />
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ConfigField({ label, description, value, onSave, saving }: {
  label: string; description?: string; value: unknown;
  onSave: (v: unknown) => void; saving: boolean
}) {
  const strValue = typeof value === 'string' && value.startsWith('****') ? value : undefined
  const isSecret = !!strValue
  const isBool = typeof value === 'boolean'

  const [editValue, setEditValue] = useState(String(value ?? ''))
  const changed = editValue !== String(value ?? '')

  const handleSave = () => {
    if (isSecret) return
    let parsed: unknown = editValue
    if (editValue === 'true') parsed = true
    else if (editValue === 'false') parsed = false
    else if (!isNaN(Number(editValue)) && editValue !== '') parsed = Number(editValue)
    onSave(parsed)
  }

  const toggleBool = () => {
    const newVal = value === true ? false : true
    onSave(newVal)
  }

  return (
    <div className="flex items-center gap-4 py-3 px-2 rounded-lg transition-colors"
      onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-bg-hover)')}
      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>{label}</div>
        {description && <div className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>{description}</div>}
      </div>
      <div className="flex items-center gap-2">
        {isBool ? (
          <button onClick={toggleBool} disabled={saving}
            className="relative w-10 h-5 rounded-full transition-colors cursor-pointer"
            style={{ background: value ? 'rgba(0,212,255,0.3)' : 'rgba(255,255,255,0.08)' }}>
            <div className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
              style={{
                left: value ? '22px' : '2px',
                background: value ? 'var(--color-accent)' : 'var(--color-text-muted)',
              }} />
          </button>
        ) : (
          <>
            <input
              value={isSecret ? strValue : editValue}
              onChange={e => setEditValue(e.target.value)}
              disabled={isSecret || saving}
              className="input-field text-xs py-1.5 px-3"
              style={{
                width: '160px',
                borderRadius: '10px',
                fontSize: '0.75rem',
                borderColor: changed ? 'rgba(0,212,255,0.3)' : undefined,
                color: isSecret ? 'var(--color-text-muted)' : undefined,
              }}
            />
            {changed && !isSecret && (
              <button onClick={handleSave} disabled={saving}
                className="p-2 rounded-lg transition-all hover:scale-105"
                style={{ background: 'var(--color-accent-glow-strong)' }}>
                <Save size={12} style={{ color: 'var(--color-accent)' }} />
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}
