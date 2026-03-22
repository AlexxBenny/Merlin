import { useEffect, useRef, useState } from 'react'
import { Send, Plus, Sparkles } from 'lucide-react'
import { api, type ChatMessage, streamChat } from '../lib/api'

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamText, setStreamText] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.getChatHistory().then(d => setMessages(d.messages || [])).catch(() => {})
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, streamText])

  const send = async () => {
    const msg = input.trim()
    if (!msg || loading) return
    setInput(''); setLoading(true); setStreamText('')
    setMessages(p => [...p, { role: 'user', content: msg, timestamp: Date.now() / 1000 }])

    try {
      await streamChat(
        msg,
        chunk => setStreamText(p => p + ' ' + chunk),
        full => { setStreamText(''); setMessages(p => [...p, { role: 'assistant', content: full, timestamp: Date.now() / 1000 }]); setLoading(false) },
        err => { setStreamText(''); setMessages(p => [...p, { role: 'assistant', content: `⚠ ${err}`, timestamp: Date.now() / 1000 }]); setLoading(false) }
      )
    } catch {
      try {
        const res = await api.chat(msg)
        setMessages(p => [...p, { role: 'assistant', content: res.response, timestamp: Date.now() / 1000 }])
      } catch (e: unknown) {
        const em = e instanceof Error ? e.message : 'Unknown error'
        setMessages(p => [...p, { role: 'assistant', content: `⚠ ${em}`, timestamp: Date.now() / 1000 }])
      }
      setLoading(false)
    }
  }

  const newSession = async () => { await api.newChatSession(); setMessages([]) }

  /* Bot avatar */
  const BotAvatar = () => (
    <div style={{
      width: 26, height: 26, borderRadius: 8, flexShrink: 0, marginTop: 3,
      background: 'linear-gradient(135deg, var(--cyan), #0090cc)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'var(--font-display)', fontSize: 10, fontWeight: 700, color: '#000',
    }}>M</div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 96px)' }}>
      {/* Actions */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <button className="btn" onClick={newSession}><Plus size={13} /> New Session</button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', paddingRight: 4 }}>
        {messages.length === 0 && !loading && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                width: 64, height: 64, borderRadius: 18, margin: '0 auto 16px',
                background: 'var(--cyan-dim)', border: '1px solid var(--border-accent)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Sparkles size={28} style={{ color: 'var(--cyan)' }} />
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-2)' }}>Ask MERLIN anything</p>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {messages.map((m, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', gap: 10 }}>
              {m.role === 'assistant' && <BotAvatar />}
              <div className={m.content.startsWith('⚠') ? 'bubble-error' : m.role === 'user' ? 'bubble-user' : 'bubble-bot'}>
                {m.content}
              </div>
            </div>
          ))}

          {/* Streaming */}
          {streamText && (
            <div style={{ display: 'flex', justifyContent: 'flex-start', gap: 10 }}>
              <BotAvatar />
              <div className="bubble-bot">
                {streamText}
                <span className="animate-bop" style={{ display: 'inline-block', width: 5, height: 14, background: 'var(--cyan)', borderRadius: 1, marginLeft: 3, verticalAlign: 'middle' }} />
              </div>
            </div>
          )}

          {/* Typing indicator */}
          {loading && !streamText && (
            <div style={{ display: 'flex', justifyContent: 'flex-start', gap: 10 }}>
              <BotAvatar />
              <div className="bubble-bot" style={{ display: 'flex', gap: 6, padding: '14px 16px' }}>
                {[0, 1, 2].map(i => <div key={i} className="typing-dot" style={{ animationDelay: `${i * 200}ms` }} />)}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input bar */}
      <div style={{ paddingTop: 14, borderTop: '1px solid var(--border)', display: 'flex', gap: 10, alignItems: 'center' }}>
        <input
          className="input"
          style={{ flex: 1 }}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder={loading ? 'Waiting for response…' : 'Ask MERLIN anything…'}
          disabled={loading}
        />
        <button className="btn btn-cyan" style={{ padding: '9px 18px' }} onClick={send} disabled={loading || !input.trim()}>
          <Send size={14} />
        </button>
      </div>
    </div>
  )
}
