import React, { useEffect, useState } from "react"
import { useNLPStore } from "@/store/nlpStore"
import { useResearchStore } from "@/store/researchStore"
import { useUIStore } from "@/store/uiStore"
import { api } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select } from "@/components/ui/select"
import { 
  FileText, 
  Upload, 
  Trash2, 
  Loader2, 
  Brain, 
  Sparkles, 
  BookOpen, 
  Clock, 
  Plus,
  AlertCircle
} from "lucide-react"

export default function Dashboard() {
  const { documents, setDocuments, removeDocument, addDocument } = useNLPStore()
  const { reports, setReports, jobs, setJobs } = useResearchStore()
  const setPage = useUIStore((state) => state.setPage)
  
  const [isUploading, setIsUploading] = useState(false)
  const [isPreloading, setIsPreloading] = useState(false)
  const [preloadKey, setPreloadKey] = useState("primary")
  const [dragActive, setDragActive] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Fetch initial stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const docs = await api.documents.list()
        setDocuments(docs)

        const pastReports = await api.research.listReports()
        setReports(pastReports)

        const pastJobs = await api.research.list()
        setJobs(pastJobs)
      } catch (err: any) {
        console.error("Failed to load dashboard data", err)
      }
    }
    fetchStats()
  }, [setDocuments, setReports, setJobs])

  const handleDeleteDoc = async (id: string) => {
    try {
      await api.documents.delete(id)
      removeDocument(id)
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to delete document.")
    }
  }

  const handlePreload = async () => {
    setIsPreloading(true)
    setErrorMsg(null)
    try {
      const docs = await api.documents.preload(preloadKey)
      setDocuments(docs)
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to preload sample corpus.")
    } finally {
      setIsPreloading(false)
    }
  }

  const handleFileUpload = async (files: FileList) => {
    setIsUploading(true)
    setErrorMsg(null)
    try {
      const fileArray = Array.from(files)
      const uploaded = await api.documents.upload(fileArray)
      uploaded.forEach((doc) => addDocument(doc))
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to upload files.")
    } finally {
      setIsUploading(false)
    }
  }

  const onDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files)
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  return (
    <div className="space-y-8 p-6">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-white/5 pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white m-0 leading-tight">Workspace Dashboard</h1>
          <p className="text-zinc-400 text-sm mt-1">Manage your research documents and view active analysis metrics.</p>
        </div>
        
        <div className="flex items-center gap-2.5">
          <Select 
            value={preloadKey} 
            onChange={(e: any) => setPreloadKey(e.target.value)}
            className="w-48"
          >
            <option value="primary">Primary Research (Text)</option>
            <option value="pdf_papers">AI Research Papers (PDF)</option>
            <option value="semantic_demo">Semantic Limitation Demo</option>
          </Select>
          <Button 
            onClick={handlePreload} 
            disabled={isPreloading}
            className="bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 text-white gap-1.5 cursor-pointer text-xs"
          >
            {isPreloading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <BookOpen className="w-3.5 h-3.5" />}
            Preload Sample
          </Button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm flex items-start gap-2">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-zinc-950/40 border-zinc-800">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-semibold text-zinc-400 uppercase tracking-wider m-0">Documents Loaded</CardTitle>
            <FileText className="w-4 h-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white">{documents.length}</div>
            <p className="text-xs text-zinc-500 mt-1">Ready for Classical NLP runs</p>
          </CardContent>
        </Card>

        <Card className="bg-zinc-950/40 border-zinc-800">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-semibold text-zinc-400 uppercase tracking-wider m-0">Compiled Reports</CardTitle>
            <Sparkles className="w-4 h-4 text-secondary" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white">{reports.length}</div>
            <p className="text-xs text-zinc-500 mt-1">Synthesized by LangGraph swarms</p>
          </CardContent>
        </Card>

        <Card className="bg-zinc-950/40 border-zinc-800">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-semibold text-zinc-400 uppercase tracking-wider m-0">Agent Runs</CardTitle>
            <Clock className="w-4 h-4 text-zinc-400" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white">{jobs.length}</div>
            <p className="text-xs text-zinc-500 mt-1">Total autonomous tasks executed</p>
          </CardContent>
        </Card>
      </div>

      {/* Workspaces CTAs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-gradient-to-br from-zinc-950 to-primary/5 border-zinc-800/80 hover:border-primary/30 transition-all duration-300 group">
          <CardHeader>
            <CardTitle className="text-xl font-bold text-white flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary group-hover:scale-110 transition-transform" />
              Classical NLP Analysis
            </CardTitle>
            <CardDescription className="text-zinc-400 mt-1.5 font-light">
              Run statistical calculations on your private corpus. Visualize document similarity heatmaps, 
              run K-Means clustering, and map topics using Latent Dirichlet Allocation.
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            <Button 
              onClick={() => setPage("nlp")} 
              className="bg-primary hover:bg-primary/95 text-white gap-1.5 cursor-pointer text-sm font-semibold"
            >
              Open NLP Workspace
              <Plus className="w-4 h-4" />
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-zinc-950 to-secondary/5 border-zinc-800/80 hover:border-secondary/30 transition-all duration-300 group">
          <CardHeader>
            <CardTitle className="text-xl font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-secondary group-hover:scale-110 transition-transform" />
              Agentic GenAI Research
            </CardTitle>
            <CardDescription className="text-zinc-400 mt-1.5 font-light">
              Deploy an autonomous research agent to decompose complex search queries, visit live URLs,
              perform FAISS semantic index chunk RAG, and generate fully cited Markdown reports.
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            <Button 
              onClick={() => setPage("research")} 
              className="bg-secondary hover:opacity-95 text-zinc-900 gap-1.5 cursor-pointer text-sm font-semibold"
            >
              Launch Research Agent
              <Sparkles className="w-4 h-4" />
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* File Manager Area */}
      <Card className="bg-zinc-950/40 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-lg font-bold text-white">Document Management</CardTitle>
          <CardDescription className="text-zinc-400">Upload private TXT or PDF documents for classical text processing.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Drag & Drop */}
          <div
            onDragEnter={onDrag}
            onDragLeave={onDrag}
            onDragOver={onDrag}
            onDrop={onDrop}
            className={`flex flex-col justify-center items-center py-10 px-4 border-2 border-dashed rounded-xl transition-all ${
              dragActive ? "border-primary bg-primary/5" : "border-zinc-800 bg-zinc-950/60 hover:bg-zinc-950"
            }`}
          >
            <Upload className="w-10 h-10 text-zinc-500 mb-3" />
            <p className="text-sm font-medium text-white">
              Drag & Drop your research documents here
            </p>
            <p className="text-xs text-zinc-500 mt-1.5 mb-4">
              Supports .pdf and .txt files up to 10MB
            </p>
            
            <label className="relative">
              <Button 
                asChild
                className="bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-white text-xs font-semibold cursor-pointer py-2 px-4 inline-flex items-center gap-1.5"
              >
                <span>
                  <Upload className="w-3.5 h-3.5" />
                  Browse Files
                </span>
              </Button>
              <input
                type="file"
                multiple
                accept=".txt,.pdf"
                onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
                className="hidden"
                disabled={isUploading}
              />
            </label>
            {isUploading && (
              <div className="flex items-center gap-1.5 text-xs text-primary mt-3">
                <Loader2 className="w-3 h-3 animate-spin" />
                Parsing and indexing files...
              </div>
            )}
          </div>

          {/* Documents List */}
          {documents.length > 0 ? (
            <div className="border border-zinc-800 rounded-lg overflow-hidden">
              <div className="max-h-[300px] overflow-y-auto">
                <table className="w-full text-left text-sm text-zinc-300">
                  <thead className="bg-zinc-900 text-zinc-400 text-xs uppercase tracking-wider font-semibold">
                    <tr>
                      <th className="px-6 py-3">Filename</th>
                      <th className="px-6 py-3">File Size</th>
                      <th className="px-6 py-3">Upload Date</th>
                      <th className="px-6 py-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800">
                    {documents.map((doc) => (
                      <tr key={doc.id} className="hover:bg-zinc-900/30 transition-colors">
                        <td className="px-6 py-4 font-medium text-white flex items-center gap-2">
                          <FileText className="w-4 h-4 text-zinc-500" />
                          {doc.filename}
                        </td>
                        <td className="px-6 py-4">{formatBytes(doc.file_size)}</td>
                        <td className="px-6 py-4">
                          {new Date(doc.created_at).toLocaleDateString(undefined, {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit"
                          })}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button
                            onClick={() => handleDeleteDoc(doc.id)}
                            className="text-zinc-500 hover:text-destructive transition-colors p-1 cursor-pointer"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-center py-10 border border-zinc-800 rounded-lg text-zinc-500 text-sm">
              No files uploaded yet. Select a sample corpus above or upload your own files to begin.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
