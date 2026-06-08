import React, { useEffect, useState } from "react"
import { useNLPStore } from "@/store/nlpStore"
import { useResearchStore } from "@/store/researchStore"
import { useUIStore } from "@/store/uiStore"
import { api } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { 
  History, 
  Brain, 
  Sparkles, 
  Trash2, 
  ArrowUpRight, 
  Clock, 
  AlertCircle
} from "lucide-react"

interface HistoryItem {
  id: string
  type: "nlp" | "agent"
  title: string
  details: string
  created_at: string
  payload: any
}

export default function HistoryTimeline() {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
  const [filterType, setFilterType] = useState<"all" | "nlp" | "agent">("all")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const setPage = useUIStore((state) => state.setPage)
  const setCurrentNLPJob = useNLPStore((state) => state.setCurrentJob)
  const setCurrentResearchJob = useResearchStore((state) => state.setCurrentJob)

  const loadHistory = async () => {
    try {
      const nlpJobs = await api.nlp.list()
      const researchJobs = await api.research.list()

      const items: HistoryItem[] = []

      nlpJobs.forEach((job) => {
        items.push({
          id: job.id,
          type: "nlp",
          title: `Classical NLP Workspace Run`,
          details: `Analyzed ${job.document_ids.length} docs (${job.parameters.k_clusters} clusters, ${job.parameters.n_topics} topics) using ${job.vectorization_mode}`,
          created_at: job.created_at,
          payload: job
        })
      })

      researchJobs.forEach((job) => {
        items.push({
          id: job.id,
          type: "agent",
          title: `Research Swarm: ${job.query}`,
          details: `Status: ${job.status.toUpperCase()} (${job.revision_count} revisions, ${job.scraped_urls.length} sites index)`,
          created_at: job.created_at,
          payload: job
        })
      })

      items.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      setHistoryItems(items)
    } catch (err) {
      console.error("Failed to load history list", err)
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  const handleDeleteItem = async (item: HistoryItem, e: React.MouseEvent) => {
    e.stopPropagation()
    setErrorMsg(null)
    try {
      if (item.type === "nlp") {
        setErrorMsg("Deleting Classical NLP runs from historical logs is not supported. Only research agent jobs can be removed.")
      } else {
        await api.research.delete(item.id)
        loadHistory()
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to remove history item.")
    }
  }

  const handleOpenItem = (item: HistoryItem) => {
    if (item.type === "nlp") {
      setCurrentNLPJob(item.payload)
      setPage("nlp")
    } else {
      setCurrentResearchJob(item.payload)
      setPage("research")
    }
  }

  const filteredItems = historyItems.filter((item) => {
    if (filterType === "all") return true
    return item.type === filterType
  })

  return (
    <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-8 space-y-6">
      {/* Background glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[450px] h-[450px] bg-blue-500/5 dark:bg-blue-600/5 rounded-full blur-[100px] pointer-events-none -z-10" />

      {/* Top filter header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-white m-0">Run History</h1>
          <p className="text-zinc-500 dark:text-zinc-400 text-xs mt-1 break-words whitespace-normal">Review, load, or delete previous Classical NLP analyses and Swarm Research runs.</p>
        </div>
        
        {/* Filtering */}
        <div className="flex gap-1 p-1 bg-zinc-100 dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800 rounded-full text-xs font-semibold">
          {[
            { id: "all", label: "All Runs" },
            { id: "nlp", label: "NLP Runs" },
            { id: "agent", label: "Agent Runs" }
          ].map((btn) => (
            <button
              key={btn.id}
              onClick={() => setFilterType(btn.id as any)}
              className={`px-3 py-1 rounded-full transition-all cursor-pointer font-bold ${
                filterType === btn.id
                  ? "bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm"
                  : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
              }`}
            >
              {btn.label}
            </button>
          ))}
        </div>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs flex items-start gap-2">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {filteredItems.length > 0 ? (
        <div className="border border-zinc-200 dark:border-zinc-800/80 rounded-2xl overflow-hidden bg-white/70 dark:bg-zinc-950/45 backdrop-blur-md shadow-sm">
          <div className="divide-y divide-zinc-200 dark:divide-zinc-900">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                onClick={() => handleOpenItem(item)}
                className="p-4 flex justify-between items-center hover:bg-zinc-50/50 dark:hover:bg-zinc-900/25 transition-all cursor-pointer group"
              >
                <div className="flex items-center gap-3.5 min-w-0">
                  <div className={`w-9 h-9 rounded-xl border flex items-center justify-center shrink-0 ${
                    item.type === "nlp" 
                      ? "bg-blue-500/10 border-blue-500/20 text-blue-600 dark:text-blue-400" 
                      : "bg-indigo-500/10 border-indigo-500/20 text-indigo-600 dark:text-indigo-400"
                  }`}>
                    {item.type === "nlp" ? <Brain className="w-4.5 h-4.5" /> : <Sparkles className="w-4.5 h-4.5" />}
                  </div>
                  
                  <div className="text-left min-w-0">
                    <h4 className="text-sm font-bold text-zinc-900 dark:text-white truncate m-0 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">{item.title}</h4>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 font-normal leading-normal truncate">{item.details}</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-xs text-zinc-400 dark:text-zinc-500 shrink-0">
                  <span className="hidden sm:flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5" />
                    {new Date(item.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </span>
                  
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      size="sm"
                      className="bg-zinc-900 hover:bg-zinc-800 dark:bg-zinc-800 dark:hover:bg-zinc-700 text-white py-1 px-3 text-xs font-bold rounded-full cursor-pointer inline-flex items-center gap-0.5"
                    >
                      Load
                      <ArrowUpRight className="w-3 h-3" />
                    </Button>
                    <button
                      onClick={(e) => handleDeleteItem(item, e)}
                      className="text-zinc-400 hover:text-destructive transition-colors p-1.5 cursor-pointer rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-900"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center border border-zinc-200 dark:border-zinc-800 border-dashed rounded-2xl p-16 text-center max-w-xl mx-auto bg-white/30 dark:bg-zinc-950/20 mt-10">
          <div className="w-12 h-12 rounded-full bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 flex items-center justify-center text-zinc-400 mb-4 animate-pulse">
            <History className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-bold text-zinc-900 dark:text-white m-0">No Runs Found</h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-500 mt-2 max-w-xs leading-relaxed font-light">
            There are no logs matching your filter. Run analyses in the workspaces to populate this history.
          </p>
        </div>
      )}
    </div>
  )
}
