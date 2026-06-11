import React, { useState, useEffect, useRef } from "react"
import { useResearchStore, ResearchJob, Report } from "@/store/researchStore"
import { useAuthStore } from "@/store/authStore"
import { api, API_BASE } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { 
  Brain, 
  Terminal, 
  Sparkles, 
  ChevronRight, 
  Loader2, 
  FileText, 
  RotateCcw,
  Clock,
  ListTodo,
  AlertCircle,
  Plus,
  ArrowRight,
  Clipboard,
  ChevronDown
} from "lucide-react"

interface WorkflowNode {
  id: string
  label: string
  desc: string
}

const WORKFLOW_NODES: WorkflowNode[] = [
  { id: "planner", label: "Planner", desc: "Decomposes Goal" },
  { id: "researcher", label: "Researcher", desc: "Searches Engine" },
  { id: "scraper", label: "Scraper", desc: "Indexes HTML" },
  { id: "synthesizer", label: "Synthesizer", desc: "Drafts Report" },
  { id: "reviewer", label: "Reviewer", desc: "Verifies Quality" }
]

export default function ResearchWorkspace() {
  const token = useAuthStore((state) => state.token)
  const user = useAuthStore((state) => state.user)
  const { 
    currentJob, 
    setCurrentJob, 
    activeNode, 
    setActiveNode, 
    logs, 
    addLog, 
    clearLogs,
    taskProgress,
    setTaskProgress,
    feedback,
    setFeedback,
    resetActiveJob
  } = useResearchStore()

  const [query, setQuery] = useState("")
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [isStarted, setIsStarted] = useState(false)
  const [progressVal, setProgressVal] = useState(0)
  const [copied, setCopied] = useState(false)
  
  const logEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  // Cleanup ws on unmount
  useEffect(() => {
    return () => {
      if (ws) ws.close()
    }
  }, [ws])

  const handleStartResearch = async (sQuery?: string) => {
    const targetQuery = sQuery || query
    if (!targetQuery.trim()) return
    
    setIsStarted(true)
    clearLogs()
    resetActiveJob()
    setProgressVal(5)
    addLog("Initializing research session...")

    try {
      const job = await api.research.create(targetQuery.trim())
      setCurrentJob(job)
      addLog(`Created Research Job: ${job.id}`)
      connectWebSocket(job.id)
    } catch (err: any) {
      addLog(`[Error] Failed to initialize job: ${err.message}`)
      setIsStarted(false)
    }
  }

  const connectWebSocket = (jobId: string) => {
    if (ws) ws.close()

    const wsProto = API_BASE.startsWith("https:") ? "wss:" : "ws:"
    const wsBase = API_BASE.replace(/^https?:\/\//i, "")
    const wsUrl = `${wsProto}//${wsBase}/research/ws/${jobId}?token=${token}`
    const socket = new WebSocket(wsUrl)

    socket.onopen = () => {
      addLog("WebSocket telemetry link active. Listening for node transitions...")
    }

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case "info":
          addLog(`[System] ${data.message}`)
          break
          
        case "node_start":
          setActiveNode(data.node)
          addLog(`[Agent] Active node -> ${data.node.toUpperCase()}: ${data.message}`)
          const nodeIndex = WORKFLOW_NODES.findIndex(n => n.id === data.node)
          if (nodeIndex !== -1) {
            setProgressVal((nodeIndex + 1) * 20 - 10)
          }
          break
          
        case "node_complete":
          addLog(`[Agent] Completed node -> ${data.node.toUpperCase()}`)
          if (data.data) {
            const progressUpdate = data.data
            setTaskProgress(progressUpdate.task_list, [])
            if (data.node === "reviewer" && progressUpdate.status !== "needs_revision") {
              setProgressVal(100)
            }
          }
          break
          
        case "reviewer_feedback":
          setFeedback(data.feedback)
          addLog(`[Verdict] REJECTED (needs revision): ${data.feedback}`)
          break
          
        case "completed":
          setActiveNode(null)
          setProgressVal(100)
          addLog("[Success] Swarm execution complete. Report generated.")
          
          setTimeout(async () => {
            try {
              const finalJob = await api.research.get(jobId)
              setCurrentJob(finalJob)
            } catch (err) {
              console.error("Failed to load final report", err)
            }
          }, 1000)
          socket.close()
          setIsStarted(false)
          break
          
        case "failed":
          setActiveNode(null)
          addLog(`[Fail] Swarm error: ${data.error}`)
          socket.close()
          setIsStarted(false)
          break
          
        default:
          break
      }
    }

    socket.onerror = () => {
      addLog("[Error] WebSocket disconnected unexpectedly.")
      setIsStarted(false)
    }

    socket.onclose = () => {
      addLog("Telemetry stream closed.")
    }

    setWs(socket)
  }

  const handleCopyReport = () => {
    if (!currentJob?.report_draft) return
    navigator.clipboard.writeText(currentJob.report_draft)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getNodeClass = (nodeId: string) => {
    const isCompleted = currentJob?.status === "completed"
    if (activeNode === nodeId) {
      return "border-blue-500 dark:border-blue-400 bg-blue-500/10 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 shadow-md shadow-blue-500/10 animate-pulse font-bold scale-102"
    }
    
    const activeIdx = WORKFLOW_NODES.findIndex(n => n.id === activeNode)
    const thisIdx = WORKFLOW_NODES.findIndex(n => n.id === nodeId)
    
    if (isCompleted || (activeIdx !== -1 && thisIdx < activeIdx)) {
      return "border-emerald-500 bg-emerald-500/5 text-emerald-600 dark:text-emerald-400"
    }
    
    return "border-zinc-200 dark:border-zinc-800/80 bg-white dark:bg-zinc-950/40 text-zinc-400 dark:text-zinc-550"
  }

  // Advanced, clean markdown formatter producing structured React components
  const parseMarkdownToReact = (text: string) => {
    if (!text) return null
    const lines = text.split("\n")
    const elements: React.ReactNode[] = []
    let currentList: { type: "ul" | "ol"; items: string[] } | null = null

    const flushList = (key: number) => {
      if (!currentList) return null
      const listType = currentList.type
      const items = currentList.items
      currentList = null

      if (listType === "ul") {
        return (
          <ul key={`list-${key}`} className="list-disc pl-5 mb-5 space-y-1.5 text-zinc-700 dark:text-zinc-300 text-sm leading-relaxed">
            {items.map((item, idx) => (
              <li key={idx} className="pl-1">{renderTextWithFormatting(item)}</li>
            ))}
          </ul>
        )
      } else {
        return (
          <ol key={`list-${key}`} className="list-decimal pl-5 mb-5 space-y-2 text-zinc-700 dark:text-zinc-300 text-sm leading-relaxed">
            {items.map((item, idx) => (
              <li key={idx} className="pl-1 break-all">{renderTextWithFormatting(item)}</li>
            ))}
          </ol>
        )
      }
    }

    const renderTextWithFormatting = (str: string) => {
      const citationRegex = /\[Source:\s*([^\]]+)\]/g
      let parts: React.ReactNode[] = []
      let lastIndex = 0
      let match
      
      const formatInline = (textSegment: string): React.ReactNode[] => {
        const boldParts = textSegment.split(/\*\*(.*?)\*\//g)
        // Standard bold parser split (alternate indexes are bold contents)
        const outputParts: React.ReactNode[] = []
        const regexBold = /\*\*(.*?)\*\*/g
        let innerLastIndex = 0
        let innerMatch
        
        while ((innerMatch = regexBold.exec(textSegment)) !== null) {
          const prevText = textSegment.substring(innerLastIndex, innerMatch.index)
          if (prevText) {
            outputParts.push(...formatUrls(prevText))
          }
          outputParts.push(
            <strong key={`bold-${innerMatch.index}`} className="font-bold text-zinc-900 dark:text-white">
              {innerMatch[1]}
            </strong>
          )
          innerLastIndex = regexBold.lastIndex
        }
        const remainingText = textSegment.substring(innerLastIndex)
        if (remainingText) {
          outputParts.push(...formatUrls(remainingText))
        }
        return outputParts
      }

      const formatUrls = (subPart: string): React.ReactNode[] => {
        const urlRegex = /(https?:\/\/[^\s]+)/g
        const urlParts = subPart.split(urlRegex)
        return urlParts.map((urlSeg, urlIdx) => {
          if (urlIdx % 2 === 1) {
            // Remove trailing punctuation from URLs if any
            let cleanUrl = urlSeg
            let trailing = ""
            if (/[.,;:)\]]$/.test(cleanUrl)) {
              trailing = cleanUrl.slice(-1)
              cleanUrl = cleanUrl.slice(0, -1)
            }
            return (
              <React.Fragment key={urlIdx}>
                <a 
                  href={cleanUrl} 
                  target="_blank" 
                  rel="noreferrer" 
                  className="text-blue-600 dark:text-blue-400 hover:underline break-all inline-block"
                >
                  {cleanUrl}
                </a>
                {trailing}
              </React.Fragment>
            )
          }
          return urlSeg
        })
      }

      while ((match = citationRegex.exec(str)) !== null) {
        const preceding = str.substring(lastIndex, match.index)
        if (preceding) {
          parts.push(...formatInline(preceding))
        }
        const url = match[1]
        parts.push(
          <a
            key={`cite-${match.index}`}
            href={url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-0.5 px-2 py-0.5 rounded bg-blue-500/10 dark:bg-blue-500/10 border border-blue-500/20 text-xs font-bold text-blue-600 dark:text-blue-400 hover:underline hover:bg-blue-500/20 transition-all ml-1 shadow-xs"
          >
            <span>Cite</span>
            <span className="w-1 h-1 bg-blue-600 dark:bg-blue-400 rounded-full"></span>
          </a>
        )
        lastIndex = citationRegex.lastIndex
      }
      
      const remaining = str.substring(lastIndex)
      if (remaining) {
        parts.push(...formatInline(remaining))
      }
      
      return parts.length > 0 ? parts : str
    }

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim()
      
      if (line.startsWith("# ")) {
        const flushed = flushList(i)
        if (flushed) elements.push(flushed)
        elements.push(
          <h1 key={i} className="text-2xl font-bold text-zinc-900 dark:text-white mt-8 mb-4 border-b border-zinc-200 dark:border-zinc-800 pb-2 font-sans">
            {renderTextWithFormatting(line.substring(2))}
          </h1>
        )
      } else if (line.startsWith("## ")) {
        const flushed = flushList(i)
        if (flushed) elements.push(flushed)
        elements.push(
          <h2 key={i} className="text-lg font-bold text-zinc-900 dark:text-white mt-6 mb-3 font-sans">
            {renderTextWithFormatting(line.substring(3))}
          </h2>
        )
      } else if (line.startsWith("### ")) {
        const flushed = flushList(i)
        if (flushed) elements.push(flushed)
        elements.push(
          <h3 key={i} className="text-sm font-bold text-zinc-800 dark:text-zinc-200 mt-4 mb-2 font-sans">
            {renderTextWithFormatting(line.substring(4))}
          </h3>
        )
      } else if (line.startsWith("- ") || line.startsWith("* ")) {
        if (!currentList) {
          currentList = { type: "ul", items: [] }
        } else if (currentList.type !== "ul") {
          const flushed = flushList(i)
          if (flushed) elements.push(flushed)
          currentList = { type: "ul", items: [] }
        }
        currentList.items.push(line.substring(2))
      } else if (/^\d+\.\s+/.test(line)) {
        const content = line.replace(/^\d+\.\s+/, "")
        if (!currentList) {
          currentList = { type: "ol", items: [] }
        } else if (currentList.type !== "ol") {
          const flushed = flushList(i)
          if (flushed) elements.push(flushed)
          currentList = { type: "ol", items: [] }
        }
        currentList.items.push(content)
      } else if (line === "") {
        const flushed = flushList(i)
        if (flushed) elements.push(flushed)
      } else {
        const flushed = flushList(i)
        if (flushed) elements.push(flushed)
        elements.push(
          <p key={i} className="my-3 text-zinc-650 dark:text-zinc-400 leading-relaxed text-sm font-normal">
            {renderTextWithFormatting(line)}
          </p>
        )
      }
    }

    const flushed = flushList(lines.length)
    if (flushed) elements.push(flushed)

    return <div className="space-y-1">{elements}</div>
  }

  const showChatInput = !currentJob || currentJob.status === "failed"

  return (
    <div className="flex-1 flex flex-col justify-center max-w-5xl mx-auto w-full px-4 py-8 relative">
      {/* Centered blue radial gradient glow behind input area (Gemini style) */}
      {showChatInput && (
        <div className="absolute top-[48%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[350px] bg-gradient-to-r from-blue-500/18 via-blue-400/12 to-indigo-500/10 dark:from-blue-600/10 dark:to-indigo-600/5 rounded-full blur-[110px] pointer-events-none -z-10 animate-pulse" />
      )}

      {/* 1. Initial Centered Chat Input (Gemini style) */}
      {showChatInput && (
        <div className="flex-1 flex flex-col items-center justify-center py-20 text-center">
          
          <h1 className="text-3xl sm:text-5xl font-medium tracking-tight text-zinc-800 dark:text-zinc-200 mb-8 leading-[1.2] max-w-2xl font-sans">
            Hi {user?.username || (user?.email ? user.email.split("@")[0] : "Researcher")}, what you wanna research on?
          </h1>

          {/* Search Pill Input */}
          <div className="relative max-w-2xl w-full shadow-xl shadow-blue-500/5 rounded-full border border-zinc-200/80 dark:border-zinc-800 bg-white/90 dark:bg-zinc-900/80 backdrop-blur-md focus-within:ring-2 focus-within:ring-blue-500/20 transition-all flex items-center p-2">
            
            <input
              type="text"
              placeholder="Ask Swarm..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleStartResearch()}
              disabled={isStarted}
              className="flex-1 bg-transparent border-0 outline-none pl-5 pr-2 text-zinc-850 dark:text-zinc-100 placeholder:text-zinc-400 dark:placeholder:text-zinc-500 text-sm py-2"
            />
            
            <button
              onClick={() => handleStartResearch()}
              disabled={isStarted || !query.trim()}
              className="h-10 px-5 rounded-full bg-blue-600 hover:bg-blue-550 text-white font-bold text-xs flex items-center gap-1 cursor-pointer transition-colors shrink-0 disabled:opacity-50 shadow-sm shadow-blue-500/15"
            >
              {isStarted ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <>
                  <span>Research</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>

          {/* Suggestions pills */}
          <div className="flex flex-wrap gap-2.5 justify-center mt-8 max-w-xl">
            {[
              "Trapped-ion vs Superconducting Qubits coherence",
              "Shor algorithm security implications on RSA",
              "State space models vs Transformers in 2026",
              "Zero-trust microsegmentation architecture"
            ].map((sQuery, i) => (
              <button
                key={i}
                onClick={() => {
                  setQuery(sQuery)
                  handleStartResearch(sQuery)
                }}
                className="px-3.5 py-1.5 rounded-full border border-zinc-200 dark:border-zinc-850 bg-white/40 dark:bg-zinc-900/30 hover:border-blue-500/30 hover:bg-blue-500/5 dark:hover:bg-blue-600/5 transition-all text-xs font-semibold text-zinc-500 dark:text-zinc-450 hover:text-zinc-800 dark:hover:text-white cursor-pointer shadow-sm"
              >
                {sQuery}
              </button>
            ))}
          </div>

        </div>
      )}

      {/* 2. Swarm execution & Result layout */}
      {currentJob && currentJob.status !== "failed" && (
        <div className="flex-1 flex flex-col space-y-6 animate-fade-in py-4">
          
          {/* Floating compact header */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-white/50 dark:bg-zinc-950/35 border border-zinc-200 dark:border-zinc-800/80 p-4 rounded-2xl backdrop-blur-md">
            <div className="min-w-0">
              <span className="text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider block">Active Swarm Objective</span>
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 italic truncate max-w-xl mt-0.5">"{currentJob.query}"</p>
            </div>
            
            {!isStarted && (currentJob.status === "completed" || currentJob.status === "failed") && (
              <Button 
                onClick={resetActiveJob}
                className="bg-zinc-900 dark:bg-zinc-800 hover:bg-zinc-800 dark:hover:bg-zinc-700 text-white text-xs font-semibold py-1.5 px-4 cursor-pointer inline-flex items-center gap-1.5 rounded-full"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                New Inquiry
              </Button>
            )}
          </div>

          {/* Progress Bar */}
          {currentJob.status === "running" && (
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider">
                <span>Swarm telemetry compilation</span>
                <span className="text-blue-600 dark:text-blue-400">{progressVal}%</span>
              </div>
              <Progress value={progressVal} className="h-1 bg-zinc-200 dark:bg-zinc-800" />
            </div>
          )}

          {/* Live Node Graph Map (Emojis/Icons Removed) */}
          <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
            <CardHeader className="pb-3 border-b border-zinc-100 dark:border-zinc-900">
              <CardTitle className="text-sm font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider flex items-center gap-1.5 m-0">
                <Brain className="w-3.5 h-3.5 text-blue-500" />
                LangGraph State Machine Swarm Map
              </CardTitle>
            </CardHeader>
            <CardContent className="py-5 overflow-x-auto flex justify-center">
              <div className="flex flex-col sm:flex-row items-center gap-3 sm:gap-2">
                {WORKFLOW_NODES.map((node, idx) => (
                  <React.Fragment key={node.id}>
                    <div className={`w-36 p-3.5 rounded-xl border flex flex-col justify-center items-center text-center transition-all duration-300 ${getNodeClass(node.id)}`}>
                      <span className="text-xs font-bold block leading-none truncate">{node.label}</span>
                      <span className="text-xs text-zinc-450 dark:text-zinc-550 block mt-1.5 leading-none truncate">{node.desc}</span>
                    </div>

                    {idx < WORKFLOW_NODES.length - 1 && (
                      <div className="h-4 w-4 flex items-center justify-center text-zinc-300 dark:text-zinc-800 rotate-90 sm:rotate-0">
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* 3. Restructured Panels based on Job State */}
          
          {/* Active Running State: Logs & Task Decompositions (2/3 + 1/3) */}
          {currentJob.status === "running" && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
              
              {/* Telemetry Output Terminal (2/3 width) */}
              <div className="lg:col-span-2 space-y-6">
                <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                  <CardHeader className="pb-3 border-b border-zinc-100 dark:border-zinc-900">
                    <CardTitle className="text-sm font-bold text-zinc-450 dark:text-zinc-550 uppercase tracking-wider flex items-center gap-1.5 m-0">
                      <Terminal className="w-3.5 h-3.5 text-zinc-400" />
                      Live Swarm Telemetry Log
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-3">
                    <div className="h-72 bg-zinc-100 dark:bg-zinc-950 p-3.5 rounded-lg border border-zinc-200 dark:border-zinc-900 font-mono text-xs overflow-y-auto space-y-1.5 flex flex-col">
                      {logs.map((log, i) => (
                        <div key={i} className="leading-relaxed break-all">
                          <span className="text-zinc-450 dark:text-zinc-650 mr-1.5">
                            {new Date().toLocaleTimeString(undefined, { hour12: false })}
                          </span>
                          <span className="text-zinc-800 dark:text-green-400 font-semibold">{log}</span>
                        </div>
                      ))}
                      <div ref={logEndRef} />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Sub-Goals Decomposed (1/3 width) */}
              <div className="lg:col-span-1 space-y-6">
                {feedback && (
                  <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 text-xs leading-normal space-y-1 shadow-sm flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold block">Reviewer Feedback</span>
                      <p className="font-normal italic mt-0.5">"{feedback}"</p>
                    </div>
                  </div>
                )}

                {currentJob.task_list.length > 0 && (
                  <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                    <CardHeader className="pb-3 border-b border-zinc-100 dark:border-zinc-900">
                      <CardTitle className="text-sm font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider flex items-center gap-1.5 m-0">
                        <ListTodo className="w-3.5 h-3.5 text-blue-500" />
                        Decomposed Query Sub-Goals
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-3 space-y-2 max-h-[280px] overflow-y-auto">
                      {currentJob.task_list.map((task, i) => (
                        <div key={i} className="flex items-start gap-2 p-2 rounded-lg bg-zinc-50 dark:bg-zinc-900/40 border border-zinc-150 dark:border-zinc-855 text-xs text-zinc-650 dark:text-zinc-400 leading-normal">
                          <span className="font-bold text-blue-600 dark:text-blue-400 shrink-0">{i+1}.</span>
                          <span className="font-normal">{task}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}
              </div>

            </div>
          )}

          {/* Completed State: Full-Width Report Output */}
          {currentJob.status === "completed" && currentJob.report && (
            <div className="space-y-6">
              
              {/* Full-Width Report Content */}
              <Card className="w-full bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                <CardHeader className="p-4 border-b border-zinc-150 dark:border-zinc-900 flex flex-row justify-between items-center">
                  <div className="min-w-0">
                    <CardTitle className="text-sm font-bold text-zinc-900 dark:text-white m-0">Compiled Swarm Report</CardTitle>
                    <CardDescription className="text-zinc-450 mt-1 text-xs">Autonomous synthesis finalized and certified by Reviewer Agent.</CardDescription>
                  </div>
                  <Button
                    size="sm"
                    onClick={handleCopyReport}
                    className="bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-900 dark:hover:bg-zinc-800 text-zinc-850 dark:text-zinc-200 text-xs border border-zinc-200 dark:border-zinc-800 rounded-full h-8 px-4 flex items-center gap-1.5 cursor-pointer transition-all active:scale-[0.97] shrink-0"
                  >
                    <Clipboard className="w-3.5 h-3.5" />
                    <span>{copied ? "Copied" : "Copy"}</span>
                  </Button>
                </CardHeader>
                <CardContent className="p-6 sm:p-10">
                  <div className="prose dark:prose-invert max-w-none text-zinc-750 dark:text-zinc-350 select-text">
                    {parseMarkdownToReact(currentJob.report.content)}
                  </div>
                </CardContent>
              </Card>

              {/* Full-Width Metrics Row */}
              <Card className="w-full bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                <CardHeader className="pb-3 border-b border-zinc-100 dark:border-zinc-900">
                  <CardTitle className="text-sm font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider flex items-center gap-1.5 m-0">
                    <Clock className="w-3.5 h-3.5 text-zinc-400" />
                    Execution Performance Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-2 sm:grid-cols-5 gap-3 p-3">
                  {[
                    { val: currentJob.report.metrics.tasks, label: "Sub-Queries" },
                    { val: currentJob.report.metrics.sources, label: "Web Sources" },
                    { val: currentJob.report.metrics.pages_scraped, label: "Pages Indexed" },
                    { val: currentJob.report.metrics.context_chunks, label: "FAISS Nodes" },
                    { val: currentJob.report.metrics.revisions, label: "Review Cycles" }
                  ].map((stat, i) => (
                    <div key={i} className="p-3.5 bg-zinc-50 dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-850 rounded-xl text-center shadow-xs">
                      <span className="text-xl font-bold text-zinc-800 dark:text-white block leading-none">{stat.val}</span>
                      <span className="text-xs font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mt-2 block">{stat.label}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Collapsible Telemetry & Decomposed Subgoals Drawer */}
              <details className="group border border-zinc-200 dark:border-zinc-800 rounded-2xl overflow-hidden bg-white/40 dark:bg-zinc-950/20 backdrop-blur-md">
                <summary className="flex items-center justify-between p-4 cursor-pointer select-none text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider bg-zinc-50/50 dark:bg-zinc-950/30">
                  <span className="flex items-center gap-1.5">
                    <Terminal className="w-4 h-4" />
                    Show Swarm Execution Telemetry & Logs
                  </span>
                  <ChevronDown className="w-4 h-4 text-blue-500 group-open:rotate-180 transition-transform" />
                </summary>
                
                <div className="p-4 grid grid-cols-1 md:grid-cols-3 gap-6 border-t border-zinc-200 dark:border-zinc-850">
                  {/* Telemetry Console (2/3 width) */}
                  <div className="md:col-span-2 space-y-4">
                    <span className="text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider block">Live Output History</span>
                    <div className="h-60 bg-zinc-100 dark:bg-zinc-950 p-3.5 rounded-lg border border-zinc-200 dark:border-zinc-900 font-mono text-xs overflow-y-auto space-y-1.5 flex flex-col">
                      {logs.map((log, i) => (
                        <div key={i} className="leading-relaxed break-all">
                          <span className="text-zinc-455 dark:text-zinc-650 mr-1.5">
                            {new Date().toLocaleTimeString(undefined, { hour12: false })}
                          </span>
                          <span className="text-zinc-800 dark:text-green-400 font-medium">{log}</span>
                        </div>
                      ))}
                      <div ref={logEndRef} />
                    </div>
                  </div>

                  {/* Subgoals list (1/3 width) */}
                  <div className="md:col-span-1 space-y-4">
                    <span className="text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider block">Query Sub-questions</span>
                    <div className="space-y-2 max-h-[240px] overflow-y-auto pr-1">
                      {currentJob.task_list.map((task, i) => (
                        <div key={i} className="flex items-start gap-2 p-2 rounded-lg bg-zinc-50 dark:bg-zinc-900/40 border border-zinc-150 dark:border-zinc-850 text-xs text-zinc-650 dark:text-zinc-400 leading-normal">
                          <span className="font-bold text-blue-600 dark:text-blue-400 shrink-0">{i+1}.</span>
                          <span className="font-normal">{task}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </details>

            </div>
          )}

        </div>
      )}

    </div>
  )
}
