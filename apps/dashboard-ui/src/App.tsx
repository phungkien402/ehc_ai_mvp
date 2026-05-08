import { useEffect, useMemo, useRef, useState } from 'react'
import mermaid from 'mermaid'

// ── Types ─────────────────────────────────────────────────────────────────────
type TabKey = 'overview' | 'logs' | 'workflow' | 'query' | 'system'

type WorkflowNodeId =
  | 'extract_ocr_if_image'
  | 'agent'
  | 'should_continue'
  | 'tools'
  | 'grade_documents'
  | 'route_after_grading'
  | 'rewrite_query'
  | 'generate_final_answer'

type Overview = {
  api_status: string
  uptime_seconds: number
  qdrant_ok: boolean
  ollama_ok: boolean
  active_models: string[]
  recent_latency_ms: number | null
}

type LogEntry = {
  cursor: number
  timestamp: string
  logger: string
  level: string
  message: string
  raw: string
}

type LogsResponse = {
  entries: LogEntry[]
  next_cursor: number
}

type WorkflowPayload = {
  content: string
  last_modified_epoch_ms: number
}

type WorkflowEvent = {
  key: string
  nodeId: WorkflowNodeId
  phase?: string
  traceId?: string
  timestamp: string
  message: string
  level: string
}

type GpuInfo = {
  index: number
  name: string
  util_percent: number
  mem_used_mb: number
  mem_total_mb: number
  mem_percent: number
  temp_c: number | null
  power_w: number | null
  power_limit_w: number | null
}

type SystemStats = {
  cpu_percent: number
  cpu_count: number
  cpu_freq_mhz: number | null
  ram_total_gb: number
  ram_used_gb: number
  ram_percent: number
  swap_total_gb: number
  swap_used_gb: number
  disk_total_gb: number
  disk_used_gb: number
  disk_percent: number
  net_sent_mb: number
  net_recv_mb: number
  load_avg: number[]
  gpus: GpuInfo[]
}

type QueryResult = {
  answer: string
  ocr_text?: string
  sources: { issue_id: string; source_title?: string; snippet: string; url: string; score: number; source_type?: string }[]
  error: string | null
  execution_time_ms: number
}

// ── Constants ─────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_BASE || ''
const MAX_LOG_BUFFER = 3000
const WORKFLOW_SCAN_WINDOW = 500
const WORKFLOW_TIMELINE_LIMIT = 20

const WORKFLOW_NODE_LABEL: Record<WorkflowNodeId, string> = {
  extract_ocr_if_image: 'OCR',
  agent: 'Agent',
  should_continue: 'Router',
  tools: 'Retrieval',
  grade_documents: 'Grader',
  route_after_grading: 'Post-grade',
  rewrite_query: 'Rewrite',
  generate_final_answer: 'Generate',
}

const NODE_COLOR: Record<WorkflowNodeId, string> = {
  extract_ocr_if_image: '#818cf8',
  agent: '#22d3ee',
  should_continue: '#94a3b8',
  tools: '#34d399',
  grade_documents: '#fbbf24',
  route_after_grading: '#f87171',
  rewrite_query: '#a78bfa',
  generate_final_answer: '#4ade80',
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function parseTraceParts(msg: string): Record<string, string> | null {
  if (!msg.startsWith('TRACE|')) return null
  const out: Record<string, string> = {}
  for (const seg of msg.slice(6).split('|')) {
    const eq = seg.indexOf('=')
    if (eq > 0) out[seg.slice(0, eq)] = seg.slice(eq + 1)
  }
  return out
}

function formatUptime(sec: number): string {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = sec % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function getWsUrl(src: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/api/v1/dashboard/logs/stream?log_source=${encodeURIComponent(src)}`
}

function detectWorkflowEvent(entry: LogEntry): WorkflowEvent | null {
  const trace = parseTraceParts(entry.message)
  if (trace) {
    const nodeId = trace.node as WorkflowNodeId
    if (nodeId in WORKFLOW_NODE_LABEL) {
      const detail = Object.entries(trace)
        .filter(([k]) => !['node', 'phase', 'trace_id'].includes(k))
        .map(([k, v]) => `${k}=${v}`)
        .join(' · ')
      return {
        key: `${entry.timestamp}|${entry.message}`,
        nodeId,
        phase: trace.phase,
        traceId: trace.trace_id,
        timestamp: entry.timestamp,
        message: detail || trace.phase || '',
        level: entry.level,
      }
    }
  }
  const msg = entry.message.toLowerCase()
  const logger = entry.logger.toLowerCase()
  const make = (id: WorkflowNodeId): WorkflowEvent => ({
    key: `${entry.timestamp}|${entry.message}`,
    nodeId: id,
    timestamp: entry.timestamp,
    message: entry.message,
    level: entry.level,
  })
  if (msg.includes('no image; skipping ocr') || msg.includes('calling vision model') || msg.includes('ocr result')) return make('extract_ocr_if_image')
  if (msg.includes('embedded query') || msg.includes('retrieved') || msg.includes('retrieval failed')) return make('tools')
  if (logger.includes('app.graph.router') && msg.includes('agent requested')) return make('should_continue')
  if (logger.includes('app.graph.router') && (msg.includes('document grader') || msg.includes('rewrite budget'))) return make('route_after_grading')
  if (msg.includes('agent generated')) return make('agent')
  if (msg.includes('rewrite_query failed') || msg.includes('thu lai retrieval')) return make('rewrite_query')
  if (msg.includes('synthesized answer') || msg.includes('final answer falling')) return make('generate_final_answer')
  return null
}

// ── Hooks ─────────────────────────────────────────────────────────────────────
function useClock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

// ── Components ────────────────────────────────────────────────────────────────

function UsageBar({ label, percent, detail, color }: { label: string; percent: number; detail?: string; color?: string }) {
  const pct = Math.min(100, Math.max(0, percent))
  const c = color ?? (pct > 90 ? 'var(--danger)' : pct > 70 ? 'var(--warn)' : 'var(--brand)')
  return (
    <div className="usage-bar-wrap">
      <div className="usage-bar-header">
        <span className="usage-bar-label">{label}</span>
        <span className="usage-bar-detail">{detail ?? `${pct.toFixed(1)}%`}</span>
      </div>
      <div className="usage-bar-track">
        <div className="usage-bar-fill" style={{ width: `${pct}%`, background: c }} />
      </div>
    </div>
  )
}

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const memPct = gpu.mem_percent
  const utilColor = gpu.util_percent > 90 ? 'var(--danger)' : gpu.util_percent > 70 ? 'var(--warn)' : 'var(--ok)'
  const memColor  = memPct > 90 ? 'var(--danger)' : memPct > 75 ? 'var(--warn)' : 'var(--brand)'
  const tempColor = (gpu.temp_c ?? 0) > 85 ? 'var(--danger)' : (gpu.temp_c ?? 0) > 75 ? 'var(--warn)' : 'var(--muted-2)'
  return (
    <div className="gpu-card">
      <div className="gpu-header">
        <span className="gpu-index">GPU {gpu.index}</span>
        <span className="gpu-name">{gpu.name}</span>
      </div>
      <div className="gpu-metrics">
        <div className="gpu-metric">
          <div className="gpu-ring-wrap">
            <svg viewBox="0 0 36 36" className="gpu-ring">
              <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--border-2)" strokeWidth="2.5" />
              <circle cx="18" cy="18" r="15.9" fill="none" stroke={utilColor} strokeWidth="2.5"
                strokeDasharray={`${gpu.util_percent} ${100 - gpu.util_percent}`}
                strokeDashoffset="25" strokeLinecap="round" />
            </svg>
            <span className="gpu-ring-label" style={{ color: utilColor }}>{gpu.util_percent.toFixed(0)}%</span>
          </div>
          <div className="gpu-metric-name">Compute</div>
        </div>
        <div className="gpu-metric">
          <div className="gpu-ring-wrap">
            <svg viewBox="0 0 36 36" className="gpu-ring">
              <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--border-2)" strokeWidth="2.5" />
              <circle cx="18" cy="18" r="15.9" fill="none" stroke={memColor} strokeWidth="2.5"
                strokeDasharray={`${memPct} ${100 - memPct}`}
                strokeDashoffset="25" strokeLinecap="round" />
            </svg>
            <span className="gpu-ring-label" style={{ color: memColor }}>{memPct.toFixed(0)}%</span>
          </div>
          <div className="gpu-metric-name">VRAM</div>
        </div>
        {gpu.temp_c != null && (
          <div className="gpu-metric">
            <div className="gpu-temp" style={{ color: tempColor }}>{gpu.temp_c.toFixed(0)}°C</div>
            <div className="gpu-metric-name">Temp</div>
          </div>
        )}
        {gpu.power_w != null && (
          <div className="gpu-metric">
            <div className="gpu-power">{gpu.power_w.toFixed(0)}{gpu.power_limit_w ? `/${gpu.power_limit_w.toFixed(0)}` : ''} W</div>
            <div className="gpu-metric-name">Power</div>
          </div>
        )}
      </div>
      <div className="gpu-mem-detail">
        <UsageBar
          label={`VRAM  ${(gpu.mem_used_mb / 1024).toFixed(1)} / ${(gpu.mem_total_mb / 1024).toFixed(1)} GB`}
          percent={memPct}
          detail={`${(gpu.mem_used_mb / 1024).toFixed(1)} GB`}
          color={memColor}
        />
      </div>
    </div>
  )
}

function StatCard({ label, value, sub, ok }: { label: string; value: string; sub?: string; ok?: boolean | null }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${ok === true ? 'ok' : ok === false ? 'danger' : ''}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

function ServicePill({ label, ok }: { label: string; ok: boolean }) {
  return (
    <span className={`svc-pill ${ok ? 'ok' : 'danger'}`}>
      <span className="dot" />
      {label}
    </span>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const now = useClock()
  const [activeTab, setActiveTab] = useState<TabKey>('overview')

  // Overview
  const [overview, setOverview] = useState<Overview | null>(null)
  const [overviewErr, setOverviewErr] = useState('')

  // Logs
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [logsCursor, setLogsCursor] = useState(0)
  const [wsConnected, setWsConnected] = useState(false)
  const [logsPaused, setLogsPaused] = useState(false)
  const [logSource, setLogSource] = useState('api')
  const [logLevel, setLogLevel] = useState('')
  const [logSearch, setLogSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const logsBottomRef = useRef<HTMLDivElement | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Workflow
  const [workflowMmd, setWorkflowMmd] = useState('')
  const [workflowSvg, setWorkflowSvg] = useState('')
  const [workflowErr, setWorkflowErr] = useState('')
  const [workflowTs, setWorkflowTs] = useState<number | null>(null)
  const renderSeqRef = useRef(0)

  // System stats
  const [sysStats, setSysStats] = useState<SystemStats | null>(null)
  const [sysErr, setSysErr] = useState('')

  // Query test
  const [queryText, setQueryText] = useState('')
  const [queryImageB64, setQueryImageB64] = useState<string | null>(null)
  const [queryPreview, setQueryPreview] = useState<string | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [queryErr, setQueryErr] = useState('')
  const fileRef = useRef<HTMLInputElement | null>(null)

  // ── Derived ──────────────────────────────────────────────────────────────────
  const filteredLogs = useMemo(() => logs.filter(e => {
    if (logLevel && e.level !== logLevel) return false
    if (logSearch && !e.logger.toLowerCase().includes(logSearch.toLowerCase()) &&
        !e.message.toLowerCase().includes(logSearch.toLowerCase())) return false
    return true
  }), [logs, logLevel, logSearch])

  const workflowEvents = useMemo(() => {
    const seen = new Set<string>()
    const out: WorkflowEvent[] = []
    for (const e of logs.slice(-WORKFLOW_SCAN_WINDOW)) {
      const ev = detectWorkflowEvent(e)
      if (!ev || seen.has(ev.key)) continue
      seen.add(ev.key)
      out.push(ev)
    }
    return out
  }, [logs])

  const activeNode = workflowEvents.length ? workflowEvents[workflowEvents.length - 1].nodeId : null
  const timeline = workflowEvents.slice(-WORKFLOW_TIMELINE_LIMIT).reverse()

  // ── Effects ───────────────────────────────────────────────────────────────────
  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false, theme: 'dark', securityLevel: 'loose',
      themeVariables: {
        primaryColor: '#0e7490', primaryTextColor: '#e2e8f0',
        primaryBorderColor: '#164e63', lineColor: '#475569',
        fontFamily: 'IBM Plex Mono, monospace', background: '#0d1424',
      },
    })
  }, [])

  // Overview polling
  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/dashboard/overview`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const d = await r.json() as Overview
        if (alive) { setOverview(d); setOverviewErr('') }
      } catch (e) {
        if (alive) setOverviewErr(String(e))
      }
    }
    load()
    const id = setInterval(load, 5000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  // Workflow load
  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/dashboard/workflow`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const d = await r.json() as WorkflowPayload
        if (alive) { setWorkflowMmd(d.content); setWorkflowTs(d.last_modified_epoch_ms); setWorkflowErr('') }
      } catch (e) {
        if (alive) setWorkflowErr(String(e))
      }
    }
    load()
    return () => { alive = false }
  }, [])

  // Mermaid render
  useEffect(() => {
    if (!workflowMmd) return
    const seq = ++renderSeqRef.current
    let alive = true
    const suffix = activeNode
      ? `\nclassDef activeNode fill:#f97316,stroke:#c2410c,stroke-width:3px,color:#fff;\nclass ${activeNode} activeNode;\n`
      : ''
    mermaid.render(`wf-${Date.now()}-${seq}`, `${workflowMmd}${suffix}`)
      .then(r => { if (alive && renderSeqRef.current === seq) setWorkflowSvg(r.svg) })
      .catch(e => { if (alive) setWorkflowErr(String(e)) })
    return () => { alive = false }
  }, [workflowMmd, activeNode])

  // Initial log fetch
  useEffect(() => {
    let alive = true
    fetch(`${API_BASE}/api/v1/dashboard/logs?limit=200&log_source=${encodeURIComponent(logSource)}`)
      .then(r => r.json()).then((d: LogsResponse) => {
        if (alive) { setLogs(d.entries); setLogsCursor(d.next_cursor) }
      }).catch(() => {})
    return () => { alive = false }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  // WebSocket
  useEffect(() => {
    if (logsPaused) {
      wsRef.current?.close(); wsRef.current = null; setWsConnected(false); return
    }
    const ws = new WebSocket(getWsUrl(logSource))
    wsRef.current = ws
    ws.onopen = () => setWsConnected(true)
    ws.onclose = () => setWsConnected(false)
    ws.onerror = () => setWsConnected(false)
    ws.onmessage = ev => {
      try {
        const d = JSON.parse(ev.data) as LogEntry | { type: string }
        if ((d as { type?: string }).type === 'heartbeat') return
        setLogs(prev => {
          const next = [...prev, d as LogEntry]
          return next.length > MAX_LOG_BUFFER ? next.slice(-MAX_LOG_BUFFER) : next
        })
      } catch { /* ignore */ }
    }
    return () => { ws.close(); wsRef.current = null; setWsConnected(false) }
  }, [logsPaused, logSource])

  // HTTP poll fallback
  useEffect(() => {
    if (wsConnected || logsPaused) return
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/dashboard/logs?limit=100&cursor=${logsCursor}&log_source=${encodeURIComponent(logSource)}`)
        if (!r.ok) return
        const d = await r.json() as LogsResponse
        if (!d.entries.length) return
        setLogs(prev => {
          const next = [...prev, ...d.entries]
          return next.length > MAX_LOG_BUFFER ? next.slice(-MAX_LOG_BUFFER) : next
        })
        setLogsCursor(d.next_cursor)
      } catch { /* ignore */ }
    }
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [logsCursor, wsConnected, logsPaused, logSource])

  // Auto-scroll logs
  useEffect(() => {
    if (autoScroll && logsBottomRef.current) {
      logsBottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [filteredLogs, autoScroll])

  // System polling
  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/dashboard/system`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        if (alive) { setSysStats(await r.json()); setSysErr('') }
      } catch (e) {
        if (alive) setSysErr(String(e))
      }
    }
    load()
    const id = setInterval(load, 3000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  // ── Query submit ──────────────────────────────────────────────────────────────
  const submitQuery = async () => {
    if (queryLoading || (!queryText.trim() && !queryImageB64)) return
    setQueryLoading(true); setQueryErr(''); setQueryResult(null)
    try {
      const body: Record<string, string> = {
        query: queryText.trim() || 'Ảnh này hiển thị gì? Có lỗi gì không?',
      }
      if (queryImageB64) body.image_base64 = queryImageB64
      const r = await fetch(`${API_BASE}/api/v1/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setQueryResult(await r.json())
    } catch (e) {
      setQueryErr(String(e))
    } finally {
      setQueryLoading(false)
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className="shell">

      {/* ── Top bar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="brand-dot" />
          <span className="brand-name">EHC AI <em>Ops</em></span>
        </div>

        <nav className="topbar-nav">
          {(['overview', 'logs', 'workflow', 'system', 'query'] as TabKey[]).map(tab => (
            <button
              key={tab}
              type="button"
              className={`nav-btn ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {TAB_ICON[tab]}
              <span>{TAB_LABEL[tab]}</span>
              {tab === 'logs' && logs.length > 0 && (
                <span className="badge">{logs.length >= MAX_LOG_BUFFER ? `${MAX_LOG_BUFFER}+` : logs.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="topbar-meta">
          <span className={`ws-pill ${wsConnected ? 'ok' : 'warn'}`}>
            <span className="dot" />
            {wsConnected ? 'WS Live' : 'Polling'}
          </span>
          <span className="clock">{now.toLocaleTimeString('vi-VN', { hour12: false })}</span>
        </div>
      </header>

      {/* ── Content ── */}
      <main className="content">

        {/* ── Overview ── */}
        {activeTab === 'overview' && (
          <div className="tab-pane">
            {overviewErr && <div className="alert danger">{overviewErr}</div>}

            <div className="stat-row">
              <StatCard label="API" value={overview?.api_status ?? '…'} ok={overview ? true : null} />
              <StatCard label="Uptime" value={overview ? formatUptime(overview.uptime_seconds) : '…'} />
              <StatCard
                label="Avg Latency"
                value={overview?.recent_latency_ms != null ? `${overview.recent_latency_ms.toFixed(0)} ms` : 'N/A'}
                ok={overview?.recent_latency_ms != null ? overview.recent_latency_ms < 3000 : null}
              />
              <StatCard label="Log Buffer" value={String(logs.length)} sub={`/ ${MAX_LOG_BUFFER} max`} />
            </div>

            <div className="card">
              <div className="card-title">Services</div>
              <div className="svc-row">
                <ServicePill label="Qdrant" ok={overview?.qdrant_ok ?? false} />
                <ServicePill label="Ollama" ok={overview?.ollama_ok ?? false} />
              </div>
            </div>

            <div className="card">
              <div className="card-title">Active Models</div>
              {!overview && <div className="muted">Loading…</div>}
              {overview?.active_models.length === 0 && <div className="muted">No active models</div>}
              <ul className="model-list">
                {(overview?.active_models ?? []).map(m => <li key={m}>{m}</li>)}
              </ul>
            </div>
          </div>
        )}

        {/* ── Logs ── */}
        {activeTab === 'logs' && (
          <div className="tab-pane logs-pane">
            <div className="toolbar">
              <div className="toolbar-left">
                <select value={logSource} onChange={e => { setLogSource(e.target.value); setLogs([]); setLogsCursor(0) }}>
                  <option value="api">api.log</option>
                  <option value="agent">agent.log</option>
                  <option value="bot">bot.log</option>
                </select>
                <select value={logLevel} onChange={e => setLogLevel(e.target.value)}>
                  <option value="">All levels</option>
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARN</option>
                  <option value="ERROR">ERROR</option>
                </select>
                <input
                  value={logSearch}
                  onChange={e => setLogSearch(e.target.value)}
                  placeholder="Search logger or message…"
                  className="search-input"
                />
              </div>
              <div className="toolbar-right">
                <label className="toggle-label">
                  <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
                  Auto-scroll
                </label>
                <button type="button" className={`btn ${logsPaused ? 'btn-ok' : 'btn-warn'}`} onClick={() => setLogsPaused(v => !v)}>
                  {logsPaused ? '▶ Resume' : '⏸ Pause'}
                </button>
                <button type="button" className="btn" onClick={() => setLogs([])}>Clear</button>
              </div>
            </div>

            <div className="log-wrap">
              <table className="log-table">
                <thead>
                  <tr>
                    <th style={{ width: 145 }}>Time</th>
                    <th style={{ width: 72 }}>Level</th>
                    <th style={{ width: 220 }}>Logger</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredLogs.map((e, i) => (
                    <tr key={`${e.cursor}-${i}`} className={`log-row ${e.level}`}>
                      <td className="mono dim">{e.timestamp?.slice(11, 23) || '—'}</td>
                      <td><span className={`lvl ${e.level}`}>{e.level || 'RAW'}</span></td>
                      <td className="mono dim ellipsis" title={e.logger}>{e.logger}</td>
                      <td className="mono log-msg">{e.message}</td>
                    </tr>
                  ))}
                  {filteredLogs.length === 0 && (
                    <tr><td colSpan={4} className="empty-row">No matching log entries</td></tr>
                  )}
                </tbody>
              </table>
              <div ref={logsBottomRef} />
            </div>
          </div>
        )}

        {/* ── Workflow ── */}
        {activeTab === 'workflow' && (
          <div className="tab-pane workflow-pane">
            <div className="wf-header">
              <div className="wf-active">
                {activeNode
                  ? <><span className="dot ok" /> Running: <strong style={{ color: NODE_COLOR[activeNode] }}>{WORKFLOW_NODE_LABEL[activeNode]}</strong></>
                  : <><span className="dot muted" /> Idle</>
                }
              </div>
              <div className="muted" style={{ fontSize: '0.8rem' }}>
                {workflowTs ? `Updated ${new Date(workflowTs).toLocaleTimeString()}` : ''}
              </div>
            </div>

            {workflowErr && <div className="alert danger">{workflowErr}</div>}
            {!workflowSvg && !workflowErr && <div className="muted" style={{ padding: '2rem', textAlign: 'center' }}>Loading workflow diagram…</div>}

            {workflowSvg && (
              <div className="wf-canvas" dangerouslySetInnerHTML={{ __html: workflowSvg }} />
            )}

            <div className="wf-timeline">
              <div className="wf-timeline-title">Recent Events</div>
              {timeline.length === 0 && <div className="muted" style={{ padding: '0.75rem 1rem' }}>No workflow events in current log stream</div>}
              {timeline.map(ev => (
                <div key={ev.key} className="wf-event">
                  <span className="wf-dot" style={{ background: NODE_COLOR[ev.nodeId] }} />
                  <span className="wf-node" style={{ color: NODE_COLOR[ev.nodeId] }}>{WORKFLOW_NODE_LABEL[ev.nodeId]}</span>
                  {ev.phase && <span className={`wf-phase phase-${ev.phase}`}>{ev.phase}</span>}
                  <span className="wf-msg">{ev.traceId ? `[${ev.traceId.slice(0, 8)}] ` : ''}{ev.message}</span>
                  <span className="wf-time">{ev.timestamp?.slice(11, 19) || ''}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── System ── */}
        {activeTab === 'system' && (
          <div className="tab-pane sys-pane">
            {sysErr && <div className="alert danger">{sysErr}</div>}
            {!sysStats && !sysErr && <div className="muted" style={{ padding: '2rem', textAlign: 'center' }}>Loading system stats…</div>}

            {sysStats && (
              <>
                {/* GPU section */}
                {sysStats.gpus.length > 0 && (
                  <div className="sys-section">
                    <div className="sys-section-title">GPU · {sysStats.gpus.length} device{sysStats.gpus.length > 1 ? 's' : ''}</div>
                    <div className="gpu-grid">
                      {sysStats.gpus.map(g => <GpuCard key={g.index} gpu={g} />)}
                    </div>
                  </div>
                )}
                {sysStats.gpus.length === 0 && (
                  <div className="card">
                    <div className="card-title">GPU</div>
                    <div className="muted">No NVIDIA GPU detected (nvidia-smi not found or no GPU present)</div>
                  </div>
                )}

                {/* CPU + Memory */}
                <div className="sys-row">
                  <div className="card sys-card">
                    <div className="card-title">CPU · {sysStats.cpu_count} cores{sysStats.cpu_freq_mhz ? ` · ${(sysStats.cpu_freq_mhz / 1000).toFixed(2)} GHz` : ''}</div>
                    <UsageBar label="Utilization" percent={sysStats.cpu_percent} />
                    <div className="load-row">
                      <span className="load-label">Load avg</span>
                      {sysStats.load_avg.map((v, i) => (
                        <span key={i} className="load-val">{v.toFixed(2)}</span>
                      ))}
                      <span className="muted" style={{ fontSize: '0.72rem' }}>1m · 5m · 15m</span>
                    </div>
                  </div>

                  <div className="card sys-card">
                    <div className="card-title">Memory</div>
                    <UsageBar
                      label={`RAM  ${sysStats.ram_used_gb.toFixed(1)} / ${sysStats.ram_total_gb.toFixed(1)} GB`}
                      percent={sysStats.ram_percent}
                    />
                    {sysStats.swap_total_gb > 0 && (
                      <UsageBar
                        label={`Swap  ${sysStats.swap_used_gb.toFixed(1)} / ${sysStats.swap_total_gb.toFixed(1)} GB`}
                        percent={sysStats.swap_total_gb > 0 ? (sysStats.swap_used_gb / sysStats.swap_total_gb) * 100 : 0}
                        color="var(--warn)"
                      />
                    )}
                  </div>
                </div>

                {/* Disk + Network */}
                <div className="sys-row">
                  <div className="card sys-card">
                    <div className="card-title">Disk  /</div>
                    <UsageBar
                      label={`${sysStats.disk_used_gb.toFixed(1)} / ${sysStats.disk_total_gb.toFixed(1)} GB`}
                      percent={sysStats.disk_percent}
                    />
                  </div>

                  <div className="card sys-card">
                    <div className="card-title">Network I/O (cumulative)</div>
                    <div className="net-row">
                      <div className="net-item">
                        <span className="net-icon up">↑</span>
                        <span className="net-label">Sent</span>
                        <span className="net-val">{sysStats.net_sent_mb > 1024
                          ? `${(sysStats.net_sent_mb / 1024).toFixed(1)} GB`
                          : `${sysStats.net_sent_mb.toFixed(0)} MB`}</span>
                      </div>
                      <div className="net-item">
                        <span className="net-icon down">↓</span>
                        <span className="net-label">Recv</span>
                        <span className="net-val">{sysStats.net_recv_mb > 1024
                          ? `${(sysStats.net_recv_mb / 1024).toFixed(1)} GB`
                          : `${sysStats.net_recv_mb.toFixed(0)} MB`}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ── Query Test ── */}
        {activeTab === 'query' && (
          <div className="tab-pane query-pane">
            <div className="query-input-area">
              <div className="query-top">
                <textarea
                  className="query-text"
                  value={queryText}
                  onChange={e => setQueryText(e.target.value)}
                  placeholder="Nhập câu hỏi… (ví dụ: hướng dẫn gộp mã bệnh nhân)"
                  rows={3}
                  onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submitQuery() }}
                />
              </div>

              <div className="query-actions">
                <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }}
                  onChange={ev => {
                    const file = ev.target.files?.[0]; if (!file) return
                    const reader = new FileReader()
                    reader.onload = e => {
                      const url = e.target?.result as string
                      setQueryPreview(url); setQueryImageB64(url.split(',')[1])
                      setQueryResult(null); setQueryErr('')
                    }
                    reader.readAsDataURL(file)
                  }}
                />
                <button type="button" className="btn btn-ghost" onClick={() => fileRef.current?.click()}>
                  📎 Attach image
                </button>
                {queryImageB64 && (
                  <button type="button" className="btn btn-ghost" onClick={() => {
                    setQueryPreview(null); setQueryImageB64(null); setQueryResult(null)
                    if (fileRef.current) fileRef.current.value = ''
                  }}>✕ Remove image</button>
                )}
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={queryLoading || (!queryText.trim() && !queryImageB64)}
                  onClick={submitQuery}
                >
                  {queryLoading ? '⏳ Processing…' : '↵ Send  (Ctrl+Enter)'}
                </button>
              </div>

              {queryPreview && (
                <img src={queryPreview} alt="preview" className="query-preview" />
              )}
            </div>

            {queryErr && <div className="alert danger">{queryErr}</div>}

            {queryLoading && (
              <div className="query-loading">
                <div className="spinner" />
                <span>Waiting for AI response…</span>
              </div>
            )}

            {queryResult && (
              <div className="query-results">
                {queryResult.ocr_text && (
                  <div className="result-block">
                    <div className="result-label ocr-label">OCR — text extracted from image</div>
                    <pre className="result-pre">{queryResult.ocr_text}</pre>
                  </div>
                )}

                <div className="result-block">
                  <div className="result-label ans-label">Answer</div>
                  <div className="result-answer">{queryResult.answer}</div>
                  {queryResult.error && <div className="alert danger" style={{ marginTop: 8 }}>⚠ {queryResult.error}</div>}
                </div>

                {queryResult.sources?.length > 0 && (
                  <div className="result-block">
                    <div className="result-label src-label">Sources ({queryResult.sources.length})</div>
                    <div className="sources-list">
                      {queryResult.sources.map((s, i) => (
                        <div key={i} className="source-item">
                          <div className="source-meta">
                            {s.source_type && <span className="source-type">{s.source_type}</span>}
                            {s.url
                              ? <a href={s.url} target="_blank" rel="noreferrer" className="source-link">{s.source_title || `Ticket #${s.issue_id}`}</a>
                              : <span>{s.source_title || `Ticket #${s.issue_id}`}</span>
                            }
                            <span className="source-score">score {s.score.toFixed(3)}</span>
                          </div>
                          {s.snippet && <div className="source-snippet">{s.snippet}</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="result-timing">
                  ⏱ {queryResult.execution_time_ms?.toFixed(0) ?? '?'} ms
                </div>
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  )
}

// ── Tab metadata ───────────────────────────────────────────────────────────────
const TAB_ICON: Record<TabKey, string> = {
  overview: '◎',
  logs: '≡',
  workflow: '⊛',
  system: '⬡',
  query: '⌨',
}

const TAB_LABEL: Record<TabKey, string> = {
  overview: 'Overview',
  logs: 'Logs',
  workflow: 'Workflow',
  system: 'System',
  query: 'Query Test',
}
