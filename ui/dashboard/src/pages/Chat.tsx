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
    api.getChatHistory().then(data => setMessages(data.messages || [])).catch(() => {})
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, streamText])

  const send = async () => {
    const msg = input.trim()
    if (!msg || loading) return
    setInput('')
    setLoading(true)
    setStreamText('')

    setMessages(prev => [...prev, { role: 'user', content: msg, timestamp: Date.now() / 1000 }])

    try {
      await streamChat(
        msg,
        (chunk) => setStreamText(prev => prev + ' ' + chunk),
        (full) => {
          setStreamText('')
          setMessages(prev => [...prev, { role: 'assistant', content: full, timestamp: Date.now() / 1000 }])
          setLoading(false)
        },
        (err) => {
          setStreamText('')
          setMessages(prev => [...prev, { role: 'assistant', content: `⚠ ${err}`, timestamp: Date.now() / 1000 }])
          setLoading(false)
        }
      )
    } catch {
      try {
        const res = await api.chat(msg)
        setMessages(prev => [...prev, { role: 'assistant', content: res.response, timestamp: Date.now() / 1000 }])
      } catch (e: unknown) {
        const errMsg = e instanceof Error ? e.message : 'Unknown error'
        setMessages(prev => [...prev, { role: 'assistant', content: `⚠ ${errMsg}`, timestamp: Date.now() / 1000 }])
      }
      setLoading(false)
    }
  }

  const newSession = async () => {
    await api.newChatSession()
    setMessages([])
  }

  return (
    <div className="page-enter flex flex-col h-full" style={{ maxHeight: 'calc(100vh - 80px)' }}>
      {/* Header */}
      <div className="section-header">
        <div>
          <h1 className="section-title">Chat</h1>
          <p className="section-subtitle">Talk to MERLIN</p>
        </div>
        <button onClick={newSession} className="btn-ghost">
          <Plus size={14} /> New Session
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-auto space-y-4 mb-5 pr-1">
        {messages.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-5 animate-float"
                style={{
                  background: 'linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,92,252,0.1))',
                  border: '1px solid rgba(0,212,255,0.1)',
                }}>
                <Sparkles size={32} style={{ color: 'var(--color-accent)' }} />
              </div>
              <p className="text-sm font-medium mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                Start a conversation
              </p>
              <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Send a message to interact with MERLIN
              </p>
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} gap-3`}>
            {m.role === 'assistant' && (
              <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-1"
                style={{
                  background: 'linear-gradient(135deg, var(--color-accent), #0099cc)',
                  fontSize: '11px', fontWeight: 700, color: '#000',
                }}>
                M
              </div>
            )}
            <div className={`max-w-[65%] px-4 py-3 text-sm leading-relaxed ${
              m.role === 'user' ? 'rounded-2xl rounded-br-md' : 'rounded-2xl rounded-bl-md'
            }`}
              style={{
                background: m.role === 'user'
                  ? 'linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,180,220,0.1))'
                  : 'var(--color-bg-tertiary)',
                color: 'var(--color-text-primary)',
                border: m.role === 'user'
                  ? '1px solid rgba(0,212,255,0.15)'
                  : '1px solid var(--color-border)',
              }}>
              {m.content}
            </div>
          </div>
        ))}

        {/* Streaming */}
        {streamText && (
          <div className="flex justify-start gap-3">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-1"
              style={{
                background: 'linear-gradient(135deg, var(--color-accent), #0099cc)',
                fontSize: '11px', fontWeight: 700, color: '#000',
              }}>
              M
            </div>
            <div className="max-w-[65%] px-4 py-3 rounded-2xl rounded-bl-md text-sm leading-relaxed"
              style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text-primary)', border: '1px solid var(--color-border)' }}>
              {streamText}
              <span className="inline-block w-1.5 h-4 ml-1 rounded-sm animate-pulse" style={{ background: 'var(--color-accent)' }} />
            </div>
          </div>
        )}

        {/* Typing indicator */}
        {loading && !streamText && (
          <div className="flex justify-start gap-3">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-1"
              style={{
                background: 'linear-gradient(135deg, var(--color-accent), #0099cc)',
                fontSize: '11px', fontWeight: 700, color: '#000',
              }}>
              M
            </div>
            <div className="px-4 py-3 rounded-2xl rounded-bl-md"
              style={{ background: 'var(--color-bg-tertiary)', border: '1px solid var(--color-border)' }}>
              <div className="flex gap-1.5">
                {[0, 1, 2].map(i => (
                  <div key={i} className="typing-dot" style={{ animationDelay: `${i * 200}ms` }} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="flex gap-3">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="Ask MERLIN anything..."
          disabled={loading}
          className="input-field flex-1"
        />
        <button onClick={send} disabled={loading || !input.trim()} className="btn-primary px-5">
          <Send size={16} />
        </button>
      </div>
    </div>
  )
}
