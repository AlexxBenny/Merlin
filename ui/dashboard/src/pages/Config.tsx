import { useEffect, useState } from 'react'
import { Save } from 'lucide-react'
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
    <div className="flex items-center justify-center h-full">
      <div className="animate-pulse-glow w-3 h-3 rounded-full" style={{ background: 'var(--color-accent)' }} />
    </div>
  )

  return (
    <div className="page-enter space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>Configuration</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
            Edit MERLIN runtime settings
          </p>
        </div>
        {message && (
          <span className="text-xs px-3 py-1 rounded-full" style={{
            background: message.includes('fail') ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.15)',
            color: message.includes('fail') ? 'var(--color-error)' : 'var(--color-success)',
          }}>
            {message}
          </span>
        )}
      </div>

      {Object.entries(config).map(([section, values]) => {
        if (typeof values !== 'object' || values === null) return null
        const sectionData = values as Record<string, unknown>

        return (
          <div key={section} className="glass-card p-5">
            <h3 className="text-sm font-semibold mb-4 uppercase tracking-wider" style={{ color: 'var(--color-accent-dim)' }}>
              {section}
            </h3>
            <div className="space-y-3">
              {Object.entries(sectionData).map(([key, value]) => {
                if (typeof value === 'object' && value !== null) {
                  // Nested object
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
  const strValue = typeof value === 'string' && value.startsWith('****')
    ? value : undefined
  const isSecret = !!strValue

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

  return (
    <div className="flex items-center gap-4 py-2" style={{ borderBottom: '1px solid var(--color-border)' }}>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>{label}</div>
        {description && <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{description}</div>}
      </div>
      <div className="flex items-center gap-2">
        <input
          value={isSecret ? strValue : editValue}
          onChange={e => setEditValue(e.target.value)}
          disabled={isSecret || saving}
          className="w-40 px-3 py-1.5 rounded-lg text-xs outline-none"
          style={{
            background: 'var(--color-bg-input)',
            color: isSecret ? 'var(--color-text-muted)' : 'var(--color-text-primary)',
            border: `1px solid ${changed ? 'var(--color-accent)' : 'var(--color-border)'}`,
          }}
        />
        {changed && !isSecret && (
          <button onClick={handleSave} disabled={saving}
            className="p-1.5 rounded-md transition-colors" style={{ background: 'var(--color-accent-glow)' }}>
            <Save size={12} style={{ color: 'var(--color-accent)' }} />
          </button>
        )}
      </div>
    </div>
  )
}
