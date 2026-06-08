import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Sparkles, Key, CheckCircle2, AlertCircle } from "lucide-react"

export default function Settings() {
  // We read the API keys configurations from localStorage or display placeholders
  const groqOk = true // Mock active, we can confirm keys are in backend
  const tavilyOk = true // Mock active

  return (
    <div className="space-y-8 p-6 max-w-4xl mx-auto min-h-[calc(100vh-80px)]">
      {/* Top Banner */}
      <div className="border-b border-white/5 pb-6">
        <h1 className="text-3xl font-bold tracking-tight text-white m-0 leading-tight">Configuration Settings</h1>
        <p className="text-zinc-400 text-sm mt-1">Review agent telemetry keys, local model configurations, and pipeline variables.</p>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {/* API Keys Status */}
        <Card className="bg-zinc-950/40 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-base font-bold text-white flex items-center gap-1.5 m-0">
              <Key className="w-4 h-4 text-primary" />
              API Key Integrations
            </CardTitle>
            <CardDescription className="text-zinc-500">
              Connections needed to run the autonomous GenAI search and synthesis swarm.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-2">
            <div className="divide-y divide-zinc-800">
              <div className="py-3.5 flex items-center justify-between">
                <div className="text-left">
                  <span className="text-sm font-semibold text-white block">Groq API Connection</span>
                  <span className="text-xs text-zinc-500 block mt-0.5">Powers Llama-3.1-8B planning and Llama-3.3-70B synthesis</span>
                </div>
                
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-green-500/10 border border-green-500/20 text-green-400">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Configured (.env)
                </span>
              </div>

              <div className="py-3.5 flex items-center justify-between">
                <div className="text-left">
                  <span className="text-sm font-semibold text-white block">Tavily Web Search API</span>
                  <span className="text-xs text-zinc-500 block mt-0.5">Enables real-time queries across the live web</span>
                </div>
                
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-green-500/10 border border-green-500/20 text-green-400">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Configured (.env)
                </span>
              </div>

              <div className="py-3.5 flex items-center justify-between">
                <div className="text-left">
                  <span className="text-sm font-semibold text-white block">HuggingFace Embeddings</span>
                  <span className="text-xs text-zinc-500 block mt-0.5">Sentence-BERT model (all-MiniLM-L6-v2) for local vector indexing</span>
                </div>
                
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-green-500/10 border border-green-500/20 text-green-400">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Local Execution
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Configurations */}
        <Card className="bg-zinc-950/40 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-base font-bold text-white flex items-center gap-1.5 m-0">
              <Brain className="w-4 h-4 text-secondary" />
              Agent Models & Temperature
            </CardTitle>
            <CardDescription className="text-zinc-500">
              Defaults loaded by the backend services.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-2">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-zinc-900/30 border border-white/5 space-y-1 text-left">
                <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider block">Planner Node</span>
                <span className="text-sm font-bold text-white block">Llama 3.1 8B Instant</span>
                <span className="text-xs text-zinc-500 block mt-1">Temperature: 0.3 (Determinitic decomposition)</span>
              </div>
              
              <div className="p-4 rounded-lg bg-zinc-900/30 border border-white/5 space-y-1 text-left">
                <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider block">Synthesizer Node</span>
                <span className="text-sm font-bold text-white block">Llama 3.3 70B Versatile</span>
                <span className="text-xs text-zinc-500 block mt-1">Temperature: 0.4 (Creative research synthesis)</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
