import { useEffect, useState } from 'react'
import { Save, Settings2 } from 'lucide-react'
import { api } from '../lib/api'

export default function Config() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null)
  const [meta, setMeta] = useState<Record<string, { label: string; description: string; type: string }>>({})
  const [saving, setSaving] = useState(false)
  const [toast, setToast] = useState('')

  useEffect(() => {
    api.getConfig().then(data => {
      const fm = (data._field_metadata || {}) as typeof meta
      setMeta(fm)
      const { _field_metadata, ...rest } = data
      setConfig(rest)
    }).catch(() => {})
  }, [])

  const handleSave = async (section: string, key: string, value: unknown) => {
    setSaving(true); setToast('')
    try { const r = await api.updateConfig({ [section]: { [key]: value } }); setToast(r.message || 'Saved') }
    catch (e: unknown) { setToast(e instanceof Error ? e.message : 'Save failed') }
    setSaving(false); setTimeout(() => setToast(''), 3000)
  }

  if (!config) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '70vh' }}>
      <div className="animate-bop" style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--cyan)' }} />
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Toast */}
      {toast && (
        <div style={{ position: 'fixed', top: 20, right: 24, zIndex: 100, padding: '10px 18px', borderRadius: 10, background: 'var(--bg3)', border: '1px solid var(--border-accent)', color: 'var(--cyan)', fontSize: 13 }}>
          {toast}
        </div>
      )}

      {Object.entries(config).map(([section, values]) => {
        if (typeof values !== 'object' || values === null) return null
        const data = values as Record<string, unknown>

        return (
          <div key={section} className="card" style={{ overflow: 'hidden' }}>
            <div className="section-hdr">
              <Settings2 size={13} style={{ color: 'rgba(0,210,255,0.50)' }} />
              <span>{section}</span>
            </div>

            <div style={{ padding: '8px 10px' }}>
              {Object.entries(data).map(([key, value]) => {
                if (typeof value === 'object' && value !== null) {
                  return Object.entries(value as Record<string, unknown>).map(([sk, sv]) => (
                    <ConfigField
                      key={`${section}.${key}.${sk}`}
                      label={meta[`${section}.${key}.${sk}`]?.label || `${key}.${sk}`}
                      description={meta[`${section}.${key}.${sk}`]?.description}
                      value={sv} saving={saving}
                      onSave={v => handleSave(section, key, { [sk]: v })}
                    />
                  ))
                }
                return (
                  <ConfigField
                    key={`${section}.${key}`}
                    label={meta[`${section}.${key}`]?.label || key}
                    description={meta[`${section}.${key}`]?.description}
                    value={value} saving={saving}
                    onSave={v => handleSave(section, key, v)}
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

/* ─── Config Field ─── */
function ConfigField({ label, description, value, onSave, saving }: {
  label: string; description?: string; value: unknown; onSave: (v: unknown) => void; saving: boolean
}) {
  const isSecret = typeof value === 'string' && value.startsWith('****')
  const isBool = typeof value === 'boolean'
  const [editValue, setEditValue] = useState(String(value ?? ''))
  const changed = editValue !== String(value ?? '')

  const save = () => {
    if (isSecret) return
    let parsed: unknown = editValue
    if (editValue === 'true') parsed = true
    else if (editValue === 'false') parsed = false
    else if (!isNaN(Number(editValue)) && editValue !== '') parsed = Number(editValue)
    onSave(parsed)
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 16, padding: '10px 10px', borderRadius: 8,
      transition: 'background 0.12s',
    }}
      onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)' }}>{label}</div>
        {description && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{description}</div>}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {isBool ? (
          <div
            className={`toggle-track ${value ? 'active' : ''}`}
            onClick={() => !saving && onSave(!value)}
          >
            <div className="toggle-thumb" />
          </div>
        ) : (
          <>
            <input
              className="input"
              style={{
                width: 180, padding: '6px 12px', fontSize: 12,
                ...(changed ? { borderColor: 'rgba(0,210,255,0.35)' } : {}),
                ...(isSecret ? { color: 'var(--text-3)' } : {}),
              }}
              value={isSecret ? String(value) : editValue}
              onChange={e => setEditValue(e.target.value)}
              disabled={isSecret || saving}
            />
            {changed && !isSecret && (
              <button className="btn btn-cyan" style={{ padding: '5px 10px' }} onClick={save} disabled={saving}>
                <Save size={12} />
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}
