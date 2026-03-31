import { useEffect, useMemo, useRef, useState } from 'react'
import mermaid from 'mermaid'

type TabKey = 'overview' | 'logs' | 'workflow' | 'vision'
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
  nodeLabel: string
  timestamp: string
  message: string
  traceId?: string
  phase?: string
  logger: string
  level: string
}

const API_BASE = import.meta.env.VITE_API_BASE || ''
const MAX_LOG_BUFFER = 3000
const WORKFLOW_SCAN_WINDOW = 500
const WORKFLOW_TIMELINE_LIMIT = 20

const WORKFLOW_NODE_LABEL: Record<WorkflowNodeId, string> = {
  extract_ocr_if_image: 'extract_ocr_if_image',
  agent: 'agent',
  should_continue: 'should_continue',
  tools: 'tools',
  grade_documents: 'grade_documents',
  route_after_grading: 'route_after_grading',
  rewrite_query: 'rewrite_query',
  generate_final_answer: 'generate_final_answer',
}

function parseTraceParts(message: string): Record<string, string> | null {
  if (!message.startsWith('TRACE|')) {
    return null
  }

  const parsed: Record<string, string> = {}
  for (const segment of message.slice(6).split('|')) {
    const eq = segment.indexOf('=')
    if (eq <= 0) {
      continue
    }
    parsed[segment.slice(0, eq)] = segment.slice(eq + 1)
  }
  return parsed
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  return `${h.toString().padStart(2, '0')}:${m
    .toString()
    .padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

function getWsUrl(logSource: string): string {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const source = encodeURIComponent(logSource || 'api')
  return `${wsProtocol}//${window.location.host}/api/v1/dashboard/logs/stream?log_source=${source}`
}

function detectWorkflowEvent(entry: LogEntry): WorkflowEvent | null {
  const trace = parseTraceParts(entry.message)
  if (trace) {
    const nodeId = trace.node as WorkflowNodeId
    if (nodeId in WORKFLOW_NODE_LABEL) {
      const phase = trace.phase || 'event'
      const detail = Object.entries(trace)
        .filter(([key]) => !['node', 'phase'].includes(key))
        .map(([key, value]) => `${key}=${value}`)
        .join(' | ')

      return {
        key: `${entry.timestamp}|${entry.logger}|${entry.level}|${entry.message}`,
        nodeId,
        nodeLabel: WORKFLOW_NODE_LABEL[nodeId],
        timestamp: entry.timestamp || new Date().toLocaleString(),
        message: detail ? `${phase}: ${detail}` : phase,
        traceId: trace.trace_id,
        phase,
        logger: entry.logger,
        level: entry.level || 'RAW',
      }
    }
  }

  const msg = entry.message.toLowerCase()
  const logger = entry.logger.toLowerCase()

  const makeEvent = (nodeId: WorkflowNodeId): WorkflowEvent => ({
    key: `${entry.timestamp}|${entry.logger}|${entry.level}|${entry.message}`,
    nodeId,
    nodeLabel: WORKFLOW_NODE_LABEL[nodeId],
    timestamp: entry.timestamp || new Date().toLocaleString(),
    message: entry.message,
    logger: entry.logger,
    level: entry.level || 'RAW',
  })

  if (
    msg.includes('no image; skipping ocr') ||
    msg.includes('calling vision model for ocr') ||
    msg.includes('ocr result') ||
    msg.includes('ocr timeout') ||
    msg.includes('ocr failed')
  ) {
    return makeEvent('extract_ocr_if_image')
  }

  if (
    msg.includes('embedded query') ||
    msg.includes('retrieved') ||
    msg.includes('retrieval failed')
  ) {
    return makeEvent('tools')
  }

  if (logger.includes('app.graph.router') && msg.includes('agent requested')) {
    return makeEvent('should_continue')
  }

  if (
    logger.includes('app.graph.router') &&
    (msg.includes('document grader') || msg.includes('rewrite budget exhausted'))
  ) {
    return makeEvent('route_after_grading')
  }

  if (msg.includes('agent generated')) {
    return makeEvent('agent')
  }

  if (msg.includes('rewrite_query failed') || msg.includes('thu lai retrieval')) {
    return makeEvent('rewrite_query')
  }

  if (
    msg.includes('synthesized answer with llm') ||
    msg.includes('llm synthesis failed') ||
    msg.includes('final answer falling back')
  ) {
    return makeEvent('generate_final_answer')
  }

  return null
}

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('overview')
  const [overview, setOverview] = useState<Overview | null>(null)
  const [overviewError, setOverviewError] = useState('')

  const [logs, setLogs] = useState<LogEntry[]>([])
  const [logsCursor, setLogsCursor] = useState(0)
  const [wsConnected, setWsConnected] = useState(false)
  const [logsPaused, setLogsPaused] = useState(false)
  const [logFileSource, setLogFileSource] = useState('api')
  const [logLevelFilter, setLogLevelFilter] = useState('')
  const [logSourceFilter, setLogSourceFilter] = useState('')

  const [workflowMmd, setWorkflowMmd] = useState('')
  const [workflowError, setWorkflowError] = useState('')
  const [workflowUpdatedAt, setWorkflowUpdatedAt] = useState<number | null>(null)
  const [workflowSvg, setWorkflowSvg] = useState('')

  // Vision Test tab state
  const [visionQuery, setVisionQuery] = useState('')
  const [visionImageB64, setVisionImageB64] = useState<string | null>(null)
  const [visionImagePreview, setVisionImagePreview] = useState<string | null>(null)
  const [visionLoading, setVisionLoading] = useState(false)
  const [visionResult, setVisionResult] = useState<{
    answer: string
    ocr_text?: string
    sources: { issue_id: string; snippet: string; url: string; score: number }[]
    error: string | null
    execution_time_ms: number
  } | null>(null)
  const [visionError, setVisionError] = useState('')
  const visionFileRef = useRef<HTMLInputElement | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const renderSeqRef = useRef(0)

  const filteredLogs = useMemo(() => {
    return logs.filter((entry) => {
      if (logLevelFilter && entry.level !== logLevelFilter) {
        return false
      }
      if (
        logSourceFilter &&
        !entry.logger.toLowerCase().includes(logSourceFilter.toLowerCase())
      ) {
        return false
      }
      return true
    })
  }, [logs, logLevelFilter, logSourceFilter])

  const workflowEvents = useMemo(() => {
    const seen = new Set<string>()
    const collected: WorkflowEvent[] = []

    for (const entry of logs.slice(-WORKFLOW_SCAN_WINDOW)) {
      const event = detectWorkflowEvent(entry)
      if (!event) {
        continue
      }
      if (seen.has(event.key)) {
        continue
      }
      seen.add(event.key)
      collected.push(event)
    }

    return collected
  }, [logs])

  const activeWorkflowNode = workflowEvents.length
    ? workflowEvents[workflowEvents.length - 1].nodeId
    : null
  const workflowTimeline = workflowEvents.slice(-WORKFLOW_TIMELINE_LIMIT).reverse()

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      securityLevel: 'loose',
      themeVariables: {
        primaryColor: '#1f8f8f',
        primaryTextColor: '#101418',
        primaryBorderColor: '#0a4e4e',
        lineColor: '#24323d',
        fontFamily: 'Space Grotesk, sans-serif',
      },
    })
  }, [logFileSource])

  useEffect(() => {
    let alive = true

    const loadOverview = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/v1/dashboard/overview`)
        if (!resp.ok) {
          throw new Error(`overview http ${resp.status}`)
        }
        const data = (await resp.json()) as Overview
        if (!alive) {
          return
        }
        setOverview(data)
        setOverviewError('')
      } catch (err) {
        if (!alive) {
          return
        }
        setOverviewError(`Khong tai duoc overview: ${String(err)}`)
      }
    }

    loadOverview()
    const id = window.setInterval(loadOverview, 5000)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [])

  useEffect(() => {
    let alive = true

    const loadWorkflow = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/v1/dashboard/workflow`)
        if (!resp.ok) {
          throw new Error(`workflow http ${resp.status}`)
        }
        const payload = (await resp.json()) as WorkflowPayload
        if (!alive) {
          return
        }
        setWorkflowMmd(payload.content)
        setWorkflowUpdatedAt(payload.last_modified_epoch_ms)
        setWorkflowError('')
      } catch (err) {
        if (!alive) {
          return
        }
        setWorkflowError(`Khong tai duoc workflow: ${String(err)}`)
      }
    }

    loadWorkflow()
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    if (!workflowMmd) {
      return
    }

    const seq = renderSeqRef.current + 1
    renderSeqRef.current = seq
    let alive = true

    const dynamicClass = activeWorkflowNode
      ? `\nclassDef activeNode fill:#ffedd5,stroke:#c2410c,stroke-width:3px,color:#7c2d12;\nclass ${activeWorkflowNode} activeNode;\n`
      : ''

    mermaid
      .render(`workflow-${Date.now()}-${seq}`, `${workflowMmd}${dynamicClass}`)
      .then((rendered) => {
        if (!alive || renderSeqRef.current !== seq) {
          return
        }
        setWorkflowSvg(rendered.svg)
      })
      .catch((err) => {
        if (!alive) {
          return
        }
        setWorkflowError(`Loi render workflow: ${String(err)}`)
      })

    return () => {
      alive = false
    }
  }, [workflowMmd, activeWorkflowNode])

  useEffect(() => {
    let alive = true
    const fetchLatestLogs = async () => {
      try {
        const resp = await fetch(
          `${API_BASE}/api/v1/dashboard/logs?limit=200&log_source=${encodeURIComponent(logFileSource)}`,
        )
        if (!resp.ok) {
          throw new Error(`logs http ${resp.status}`)
        }
        const payload = (await resp.json()) as LogsResponse
        if (!alive) {
          return
        }
        setLogs(payload.entries)
        setLogsCursor(payload.next_cursor)
      } catch {
        // Silent here; websocket/polling will retry.
      }
    }

    fetchLatestLogs()
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    if (logsPaused) {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      setWsConnected(false)
      return
    }

    const ws = new WebSocket(getWsUrl(logFileSource))
    wsRef.current = ws

    ws.onopen = () => setWsConnected(true)
    ws.onclose = () => setWsConnected(false)
    ws.onerror = () => setWsConnected(false)
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as LogEntry | { type: 'heartbeat' }
        if ((data as { type?: string }).type === 'heartbeat') {
          return
        }
        const entry = data as LogEntry
        setLogs((prev) => {
          const next = [...prev, entry]
          if (next.length > MAX_LOG_BUFFER) {
            return next.slice(next.length - MAX_LOG_BUFFER)
          }
          return next
        })
      } catch {
        // ignore malformed packets
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
      setWsConnected(false)
    }
  }, [logsPaused, logFileSource])

  useEffect(() => {
    if (wsConnected || logsPaused) {
      return
    }

    const poll = async () => {
      try {
        const resp = await fetch(
          `${API_BASE}/api/v1/dashboard/logs?limit=200&cursor=${logsCursor}&log_source=${encodeURIComponent(logFileSource)}`,
        )
        if (!resp.ok) {
          return
        }
        const payload = (await resp.json()) as LogsResponse
        if (!payload.entries.length) {
          return
        }
        setLogs((prev) => {
          const next = [...prev, ...payload.entries]
          if (next.length > MAX_LOG_BUFFER) {
            return next.slice(next.length - MAX_LOG_BUFFER)
          }
          return next
        })
        setLogsCursor(payload.next_cursor)
      } catch {
        // polling fallback best-effort
      }
    }

    const id = window.setInterval(poll, 3000)
    return () => window.clearInterval(id)
  }, [logsCursor, wsConnected, logsPaused, logFileSource])

  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <h1>EHC AI Operations Console</h1>
          <p>Quan sat runtime, log stream va workflow theo thoi gian thuc.</p>
        </div>
        <div className="status-pills">
          <span className={`pill ${wsConnected ? 'ok' : 'warn'}`}>
            Logs Stream: {wsConnected ? 'WebSocket' : 'Polling'}
          </span>
          <span className="pill neutral">Cursor: {logsCursor}</span>
        </div>
      </header>

      <nav className="tabs">
        <button
          type="button"
          className={activeTab === 'overview' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          type="button"
          className={activeTab === 'logs' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('logs')}
        >
          Logs
        </button>
        <button
          type="button"
          className={activeTab === 'workflow' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('workflow')}
        >
          Visual Workflow
        </button>
        <button
          type="button"
          className={activeTab === 'vision' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('vision')}
        >
          Vision Test
        </button>
      </nav>

      {activeTab === 'overview' && (
        <section className="panel-grid">
          <article className="panel">
            <h2>System Status</h2>
            {overviewError && <p className="error-text">{overviewError}</p>}
            {!overview && !overviewError && <p>Dang tai...</p>}
            {overview && (
              <dl className="stats-list">
                <div>
                  <dt>API</dt>
                  <dd>{overview.api_status}</dd>
                </div>
                <div>
                  <dt>Uptime</dt>
                  <dd>{formatUptime(overview.uptime_seconds)}</dd>
                </div>
                <div>
                  <dt>Qdrant</dt>
                  <dd>{overview.qdrant_ok ? 'OK' : 'DOWN'}</dd>
                </div>
                <div>
                  <dt>Ollama</dt>
                  <dd>{overview.ollama_ok ? 'OK' : 'DOWN'}</dd>
                </div>
                <div>
                  <dt>Avg latency</dt>
                  <dd>
                    {overview.recent_latency_ms
                      ? `${overview.recent_latency_ms.toFixed(0)} ms`
                      : 'N/A'}
                  </dd>
                </div>
              </dl>
            )}
          </article>

          <article className="panel">
            <h2>Active Models</h2>
            <ul className="model-list">
              {(overview?.active_models || []).map((model) => (
                <li key={model}>{model}</li>
              ))}
              {overview && overview.active_models.length === 0 && (
                <li>Khong co model nao dang active</li>
              )}
            </ul>
          </article>
        </section>
      )}

      {activeTab === 'logs' && (
        <section className="panel logs-panel">
          <div className="logs-toolbar">
            <div className="toolbar-group">
              <label>
                Log File
                <select
                  value={logFileSource}
                  onChange={(e) => {
                    setLogFileSource(e.target.value)
                    setLogs([])
                    setLogsCursor(0)
                  }}
                >
                  <option value="api">api.log</option>
                  <option value="agent">agent.log</option>
                  <option value="bot">bot.log</option>
                </select>
              </label>
              <label>
                Level
                <select
                  value={logLevelFilter}
                  onChange={(e) => setLogLevelFilter(e.target.value)}
                >
                  <option value="">ALL</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                </select>
              </label>
              <label>
                Logger
                <input
                  value={logSourceFilter}
                  placeholder="app.api.routes"
                  onChange={(e) => setLogSourceFilter(e.target.value)}
                />
              </label>
            </div>
            <div className="toolbar-group">
              <button type="button" onClick={() => setLogsPaused((v) => !v)}>
                {logsPaused ? 'Resume Stream' : 'Pause Stream'}
              </button>
              <button type="button" onClick={() => setLogs([])}>
                Clear View
              </button>
            </div>
          </div>

          <div className="logs-table-wrap">
            <table className="logs-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Level</th>
                  <th>Logger</th>
                  <th>Message</th>
                </tr>
              </thead>
              <tbody>
                {filteredLogs.map((entry, idx) => (
                  <tr key={`${entry.cursor}-${idx}`}>
                    <td>{entry.timestamp || '-'}</td>
                    <td>
                      <span className={`level-badge ${entry.level || 'UNKNOWN'}`}>
                        {entry.level || 'RAW'}
                      </span>
                    </td>
                    <td>{entry.logger || '-'}</td>
                    <td>{entry.message}</td>
                  </tr>
                ))}
                {filteredLogs.length === 0 && (
                  <tr>
                    <td colSpan={4}>Khong co log phu hop bo loc.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {activeTab === 'workflow' && (
        <section className="panel workflow-panel">
          <div className="workflow-header">
            <h2>Runtime Workflow</h2>
            <div className="toolbar-group">
              <span className="workflow-active-badge ok">
                Active: {activeWorkflowNode ? WORKFLOW_NODE_LABEL[activeWorkflowNode] : 'idle'}
              </span>
              {workflowUpdatedAt && (
                <span>
                  Updated:{' '}
                  {new Date(workflowUpdatedAt).toLocaleString()}
                </span>
              )}
              <a href="/docs/workflow.png" target="_blank" rel="noreferrer">
                Open PNG
              </a>
            </div>
          </div>

          {workflowError && <p className="error-text">{workflowError}</p>}
          {!workflowError && !workflowSvg && <p>Dang tai workflow...</p>}
          {!workflowError && workflowSvg && (
            <div
              className="workflow-canvas"
              dangerouslySetInnerHTML={{ __html: workflowSvg }}
            />
          )}

          <div className="workflow-timeline">
            {workflowTimeline.map((event) => (
              <article className="workflow-timeline-item" key={event.key}>
                <span className="workflow-timeline-time">{event.timestamp || '-'}</span>
                <span className="workflow-timeline-node">{event.nodeLabel}</span>
                <span className="workflow-timeline-msg">
                  {event.traceId ? `[${event.traceId}] ` : ''}
                  {event.message}
                </span>
              </article>
            ))}
            {workflowTimeline.length === 0 && (
              <p>Chua co su kien workflow nao trong log stream.</p>
            )}
          </div>
        </section>
      )}
      {activeTab === 'vision' && (
        <section className="panel" style={{ maxWidth: 800, margin: '24px auto' }}>
          <h2>Vision Test — Thử OCR + hỏi về ảnh</h2>

          <div style={{ marginBottom: 16 }}>
            <input
              ref={visionFileRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (!file) return
                const reader = new FileReader()
                reader.onload = (ev) => {
                  const dataUrl = ev.target?.result as string
                  setVisionImagePreview(dataUrl)
                  // Strip data:image/...;base64, prefix
                  setVisionImageB64(dataUrl.split(',')[1])
                  setVisionResult(null)
                  setVisionError('')
                }
                reader.readAsDataURL(file)
              }}
            />
            <button
              type="button"
              onClick={() => visionFileRef.current?.click()}
              style={{ marginRight: 12 }}
            >
              Chọn ảnh
            </button>
            {visionImagePreview && (
              <button
                type="button"
                onClick={() => {
                  setVisionImagePreview(null)
                  setVisionImageB64(null)
                  setVisionResult(null)
                  if (visionFileRef.current) visionFileRef.current.value = ''
                }}
              >
                Xóa ảnh
              </button>
            )}
          </div>

          {visionImagePreview && (
            <img
              src={visionImagePreview}
              alt="preview"
              style={{
                maxWidth: '100%',
                maxHeight: 320,
                objectFit: 'contain',
                border: '1px solid #334',
                borderRadius: 8,
                marginBottom: 16,
                display: 'block',
              }}
            />
          )}

          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            <input
              value={visionQuery}
              onChange={(e) => setVisionQuery(e.target.value)}
              placeholder="Câu hỏi (có thể để trống nếu chỉ muốn đọc ảnh)"
              style={{ flex: 1, padding: '8px 12px', borderRadius: 6, border: '1px solid #334', background: '#101418', color: '#e2e8f0' }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !visionLoading) {
                  e.currentTarget.blur()
                  document.getElementById('vision-submit-btn')?.click()
                }
              }}
            />
            <button
              id="vision-submit-btn"
              type="button"
              disabled={visionLoading || (!visionImageB64 && !visionQuery.trim())}
              onClick={async () => {
                setVisionLoading(true)
                setVisionError('')
                setVisionResult(null)
                try {
                  const body: Record<string, string> = {
                    query: visionQuery.trim() || 'Ảnh này hiển thị gì? Có lỗi gì không?',
                  }
                  if (visionImageB64) body.image_base64 = visionImageB64
                  const resp = await fetch(`${API_BASE}/api/v1/ask`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                  })
                  const data = await resp.json()
                  setVisionResult(data)
                } catch (err) {
                  setVisionError(String(err))
                } finally {
                  setVisionLoading(false)
                }
              }}
            >
              {visionLoading ? 'Đang xử lý...' : 'Gửi'}
            </button>
          </div>

          {visionError && (
            <p className="error-text">Lỗi: {visionError}</p>
          )}

          {visionResult && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {visionResult.ocr_text && (
                <article className="panel" style={{ background: '#0d1117' }}>
                  <h3 style={{ marginTop: 0, fontSize: '0.85rem', color: '#7dd3fc' }}>OCR — Text đọc được từ ảnh</h3>
                  <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0, fontSize: '0.85rem' }}>{visionResult.ocr_text}</pre>
                </article>
              )}

              <article className="panel" style={{ background: '#0d1117' }}>
                <h3 style={{ marginTop: 0, fontSize: '0.85rem', color: '#86efac' }}>Câu trả lời</h3>
                <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{visionResult.answer}</p>
                {visionResult.error && (
                  <p style={{ marginTop: 8, color: '#fca5a5', fontSize: '0.8rem' }}>⚠ {visionResult.error}</p>
                )}
              </article>

              {visionResult.sources?.length > 0 && (
                <article className="panel" style={{ background: '#0d1117' }}>
                  <h3 style={{ marginTop: 0, fontSize: '0.85rem', color: '#c4b5fd' }}>Nguồn tham khảo</h3>
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {visionResult.sources.map((s) => (
                      <li key={s.issue_id} style={{ marginBottom: 6, fontSize: '0.85rem' }}>
                        <a href={s.url} target="_blank" rel="noreferrer" style={{ color: '#7dd3fc' }}>
                          Ticket #{s.issue_id}
                        </a>{' '}
                        <span style={{ color: '#94a3b8' }}>(score: {s.score.toFixed(3)})</span>
                        <br />
                        <span style={{ color: '#cbd5e1' }}>{s.snippet}</span>
                      </li>
                    ))}
                  </ul>
                </article>
              )}

              <p style={{ fontSize: '0.75rem', color: '#64748b', margin: 0 }}>
                Thời gian xử lý: {visionResult.execution_time_ms?.toFixed(0)} ms
              </p>
            </div>
          )}
        </section>
      )}
    </main>
  )
}

export default App
