import { useEffect, useRef, useState } from 'react'
import { Send, Plus, Sparkles, Mic, MicOff, Loader } from 'lucide-react'
import { api, type ChatMessage, type SttConfig, streamChat } from '../lib/api'

type MicState = 'idle' | 'recording' | 'processing'

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamText, setStreamText] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  // ── STT state ──
  const [micState, setMicState] = useState<MicState>('idle')
  const [sttConfig, setSttConfig] = useState<SttConfig | null>(null)
  const [sttToast, setSttToast] = useState('')
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recognitionRef = useRef<any>(null)

  useEffect(() => {
    api.getChatHistory().then(d => setMessages(d.messages || [])).catch(() => {})
    // Fetch STT config once on mount — cached for session
    api.getSttConfig().then(cfg => setSttConfig(cfg)).catch(() => {})
  }, [])

  // Re-fetch STT config when page regains focus (handles config change)
  useEffect(() => {
    const onFocus = () => {
      api.getSttConfig().then(cfg => setSttConfig(cfg)).catch(() => {})
    }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, streamText])

  const showToast = (msg: string, duration = 3000) => {
    setSttToast(msg)
    setTimeout(() => setSttToast(''), duration)
  }

  const sendMessage = async (text: string) => {
    const msg = text.trim()
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

  const send = () => sendMessage(input)

  // ── STT: Controlled mode (MediaRecorder → server) ──────

  const startControlledRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' })
      audioChunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        if (blob.size < 100) {
          setMicState('idle')
          return
        }

        setMicState('processing')
        try {
          const result = await api.sttTranscribe(blob)
          if (result.text.trim()) {
            setInput(result.text.trim())
            // Auto-send
            setMicState('idle')
            sendMessage(result.text.trim())
          } else {
            showToast('No speech detected')
            setMicState('idle')
          }
        } catch (e: unknown) {
          const msg = e instanceof Error ? e.message : 'Transcription failed'
          if (msg.includes('503')) {
            showToast('STT engine unavailable — check server config')
          } else {
            showToast(`Transcription error: ${msg}`)
          }
          setMicState('idle')
        }
      }

      recorder.onerror = () => {
        stream.getTracks().forEach(t => t.stop())
        showToast('Recording error')
        setMicState('idle')
      }

      recorder.start()
      mediaRecorderRef.current = recorder
      setMicState('recording')
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Microphone error'
      if (msg.includes('NotAllowedError') || msg.includes('Permission')) {
        showToast('Microphone permission denied')
      } else if (msg.includes('NotFoundError') || msg.includes('not found')) {
        showToast('No microphone device found')
      } else {
        showToast(`Mic error: ${msg}`)
      }
      setMicState('idle')
    }
  }

  const stopControlledRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      // State transitions to 'processing' in onstop handler
    }
  }

  // ── STT: Fast mode (Web Speech API) ────────────────────

  const startFastRecording = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) {
      // Explicit fallback with user notification
      showToast('Fast STT not supported in this browser — using server mode')
      startControlledRecording()
      return
    }

    const recognition = new SR()
    recognition.lang = 'en-US'
    recognition.interimResults = false
    recognition.maxAlternatives = 1

    recognition.onresult = (event: any) => {
      const text = event.results[0][0].transcript
      if (text.trim()) {
        setInput(text.trim())
        setMicState('idle')
        sendMessage(text.trim())
      } else {
        showToast('No speech detected')
        setMicState('idle')
      }
    }

    recognition.onerror = (event: any) => {
      if (event.error === 'not-allowed') {
        showToast('Microphone permission denied')
      } else if (event.error === 'no-speech') {
        showToast('No speech detected')
      } else if (event.error === 'network') {
        showToast('Network error — Web Speech requires internet')
      } else {
        showToast(`Speech error: ${event.error}`)
      }
      setMicState('idle')
    }

    recognition.onend = () => {
      if (micState === 'recording') setMicState('idle')
    }

    recognitionRef.current = recognition
    recognition.start()
    setMicState('recording')
  }

  const stopFastRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
      setMicState('idle')
    }
  }

  // ── Mic toggle ─────────────────────────────────────────

  const toggleMic = () => {
    if (micState === 'recording') {
      // Stop
      const mode = sttConfig?.mode || 'controlled'
      if (mode === 'fast') {
        stopFastRecording()
      } else {
        stopControlledRecording()
      }
      return
    }

    if (micState === 'processing') return // Can't toggle while processing

    // Start
    const mode = sttConfig?.mode || 'controlled'
    if (mode === 'fast') {
      startFastRecording()
    } else {
      // Controlled: check if server STT is available
      if (sttConfig && !sttConfig.available) {
        showToast('STT engine not available on server')
        return
      }
      startControlledRecording()
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

  const micDisabled = loading || micState === 'processing'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 96px)' }}>
      {/* STT Toast */}
      {sttToast && (
        <div style={{
          position: 'fixed', top: 20, right: 24, zIndex: 100,
          padding: '10px 18px', borderRadius: 10,
          background: 'var(--bg3)', border: '1px solid var(--border-accent)',
          color: 'var(--amber)', fontSize: 13,
          animation: 'fadeIn 0.2s ease',
        }}>
          {sttToast}
        </div>
      )}

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
          placeholder={
            micState === 'recording' ? 'Listening…' :
            micState === 'processing' ? 'Transcribing…' :
            loading ? 'Waiting for response…' : 'Ask MERLIN anything…'
          }
          disabled={loading || micState !== 'idle'}
        />
        <button
          className={`btn-mic ${micState === 'recording' ? 'recording' : ''} ${micState === 'processing' ? 'processing' : ''}`}
          onClick={toggleMic}
          disabled={micDisabled}
          title={
            micState === 'recording' ? 'Stop recording' :
            micState === 'processing' ? 'Transcribing…' :
            `Voice input (${sttConfig?.mode || 'controlled'} mode)`
          }
        >
          {micState === 'processing' ? (
            <Loader size={15} className="animate-spin" />
          ) : micState === 'recording' ? (
            <MicOff size={15} />
          ) : (
            <Mic size={15} />
          )}
        </button>
        <button className="btn btn-cyan" style={{ padding: '9px 18px' }} onClick={send} disabled={loading || !input.trim()}>
          <Send size={14} />
        </button>
      </div>
    </div>
  )
}
