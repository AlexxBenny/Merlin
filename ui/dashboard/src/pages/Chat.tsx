import { useEffect, useRef, useState } from 'react'
import { Send, Plus } from 'lucide-react'
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
      // Fallback to sync
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
    <div className="page-enter flex flex-col h-full" style={{ maxHeight: 'calc(100vh - 48px)' }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text-primary)' }}>Chat</h1>
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Talk to MERLIN</p>
        </div>
        <button onClick={newSession} className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors"
          style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)', border: '1px solid var(--color-border)' }}>
          <Plus size={14} /> New Session
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-auto space-y-3 mb-4 pr-2">
        {messages.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
                style={{ background: 'var(--color-accent-glow)' }}>
                <span className="text-2xl font-bold" style={{ color: 'var(--color-accent)' }}>M</span>
              </div>
              <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Send a message to start</p>
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className="max-w-[70%] px-4 py-3 rounded-2xl text-sm leading-relaxed"
              style={{
                background: m.role === 'user' ? 'var(--color-accent-dim)' : 'var(--color-bg-tertiary)',
                color: 'var(--color-text-primary)',
                borderBottomRightRadius: m.role === 'user' ? '4px' : undefined,
                borderBottomLeftRadius: m.role === 'assistant' ? '4px' : undefined,
              }}>
              {m.content}
            </div>
          </div>
        ))}

        {/* Streaming text */}
        {streamText && (
          <div className="flex justify-start">
            <div className="max-w-[70%] px-4 py-3 rounded-2xl text-sm leading-relaxed"
              style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text-primary)', borderBottomLeftRadius: '4px' }}>
              {streamText}
              <span className="inline-block w-2 h-4 ml-1 animate-pulse" style={{ background: 'var(--color-accent)' }} />
            </div>
          </div>
        )}

        {/* Loading indicator */}
        {loading && !streamText && (
          <div className="flex justify-start">
            <div className="px-4 py-3 rounded-2xl" style={{ background: 'var(--color-bg-tertiary)' }}>
              <div className="flex gap-1">
                {[0, 1, 2].map(i => (
                  <div key={i} className="w-2 h-2 rounded-full animate-bounce"
                    style={{ background: 'var(--color-accent-dim)', animationDelay: `${i * 150}ms` }} />
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
          className="flex-1 px-4 py-3 rounded-xl text-sm outline-none transition-colors"
          style={{
            background: 'var(--color-bg-input)',
            color: 'var(--color-text-primary)',
            border: '1px solid var(--color-border)',
          }}
        />
        <button onClick={send} disabled={loading || !input.trim()}
          className="px-4 py-3 rounded-xl transition-all duration-200 disabled:opacity-40"
          style={{ background: 'var(--color-accent)', color: 'var(--color-bg-primary)' }}>
          <Send size={18} />
        </button>
      </div>
    </div>
  )
}
