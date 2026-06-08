import React from "react"
import { useUIStore } from "@/store/uiStore"
import { useAuthStore } from "@/store/authStore"
import { Brain, Sparkles, ChevronRight, ShieldCheck, FileText, Search } from "lucide-react"

export default function LandingPage() {
  const setPage = useUIStore((state) => state.setPage)
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)

  const handleLaunch = (target: "nlp" | "research") => {
    setPage(isAuthenticated ? target : "auth")
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 max-w-5xl mx-auto relative overflow-hidden">
      
      {/* Background radial glows */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[700px] h-[350px] bg-gradient-to-r from-blue-500/10 to-indigo-500/10 rounded-full blur-[130px] pointer-events-none -z-10 animate-pulse" />
      <div className="absolute bottom-10 right-10 w-72 h-72 bg-blue-400/5 dark:bg-blue-600/5 rounded-full blur-[90px] pointer-events-none -z-10" />

      {/* Modern Badge */}
      <div className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-blue-500/15 bg-blue-500/5 text-blue-600 dark:text-blue-400 text-xs font-bold uppercase tracking-widest mb-8 shadow-xs">
        <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-ping" />
        <span>NLP Swarm Platform v2.0</span>
      </div>

      {/* Hero Header */}
      <div className="text-center space-y-4 max-w-3xl mb-16">
        <h1 className="text-4xl sm:text-6xl font-medium tracking-tight text-zinc-900 dark:text-white leading-[1.15] font-sans">
          The next-generation <br />
          <span className="bg-gradient-to-r from-blue-600 to-indigo-500 dark:from-blue-400 dark:to-indigo-300 bg-clip-text text-transparent font-semibold">
            NLP Research Engine
          </span>
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400 text-sm sm:text-base max-w-xl mx-auto leading-relaxed font-light">
          A dual-mode intelligence platform. Run classical statistical document mappings locally, or dispatch autonomous LLM swarms to synthesize cited research reports.
        </p>
      </div>

      {/* Main Workspaces Grid */}
      <div className="grid md:grid-cols-2 gap-8 w-full max-w-4xl">
        
        {/* Workspace Card 1: Classical NLP */}
        <div 
          onClick={() => handleLaunch("nlp")}
          className="flex flex-col p-8 rounded-2xl border border-zinc-200/60 dark:border-zinc-800/60 bg-white/40 dark:bg-zinc-950/20 backdrop-blur-md hover:border-blue-500/30 hover:shadow-xl hover:shadow-blue-500/5 hover:-translate-y-1 transition-all duration-300 group cursor-pointer"
        >
          <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-600 dark:text-blue-400 mb-6 border border-blue-500/10 group-hover:scale-105 transition-transform">
            <Brain className="w-5 h-5" />
          </div>
          
          <h3 className="text-lg font-bold text-zinc-900 dark:text-white mb-2 flex items-center gap-1.5">
            Classical NLP
            <ChevronRight className="w-4 h-4 text-zinc-400 group-hover:translate-x-1 transition-transform" />
          </h3>
          <p className="text-zinc-500 dark:text-zinc-400 text-xs font-normal leading-relaxed mb-6">
            Run local statistical clustering and matrices. Compute cosine similarity heatmaps, optimize K-Means with 2D projections, extract LDA topics, and view document sentence summaries.
          </p>

          {/* Feature highlights list */}
          <div className="space-y-2 mt-auto pt-6 border-t border-zinc-200 dark:border-zinc-900/60 text-xs font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500/50" />
              <span>Cosine Similarity Heatmaps</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500/50" />
              <span>PCA 2D Cluster Visualizations</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500/50" />
              <span>LDA Topic modeling</span>
            </div>
          </div>
        </div>

        {/* Workspace Card 2: Agentic Swarm */}
        <div 
          onClick={() => handleLaunch("research")}
          className="flex flex-col p-8 rounded-2xl border border-zinc-200/60 dark:border-zinc-800/60 bg-white/40 dark:bg-zinc-950/20 backdrop-blur-md hover:border-blue-500/30 hover:shadow-xl hover:shadow-blue-500/5 hover:-translate-y-1 transition-all duration-300 group cursor-pointer"
        >
          <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center text-indigo-650 dark:text-indigo-400 mb-6 border border-indigo-500/10 group-hover:scale-105 transition-transform">
            <Sparkles className="w-5 h-5" />
          </div>
          
          <h3 className="text-lg font-bold text-zinc-900 dark:text-white mb-2 flex items-center gap-1.5">
            Agentic Swarm
            <ChevronRight className="w-4 h-4 text-zinc-400 group-hover:translate-x-1 transition-transform" />
          </h3>
          <p className="text-zinc-500 dark:text-zinc-400 text-xs font-normal leading-relaxed mb-6">
            Deploy an autonomous multi-node agent network. Watch real-time logs as agents search, scrape web targets, index vector databases, and compile cited research reports.
          </p>

          {/* Feature highlights list */}
          <div className="space-y-2 mt-auto pt-6 border-t border-zinc-200 dark:border-zinc-900/60 text-xs font-semibold uppercase tracking-wider text-indigo-455 dark:text-zinc-500">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-550/50" />
              <span>LangGraph State Orchestration</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-550/50" />
              <span>Live Telemetry Console Streams</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-550/50" />
              <span>Fully Cited Markdown Reports</span>
            </div>
          </div>
        </div>

      </div>

      {/* Minimal Footer */}
      <div className="flex items-center gap-2 mt-20 text-zinc-400 dark:text-zinc-500 text-xs font-semibold tracking-wider uppercase">
        <ShieldCheck className="w-4 h-4 text-blue-500" />
        <span>Local SQLite & File Persistence Active</span>
      </div>

    </div>
  )
}
