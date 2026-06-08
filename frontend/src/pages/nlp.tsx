import React, { useState, useEffect } from "react"
import { useNLPStore, PCAPoint } from "@/store/nlpStore"
import { api } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Select } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  ZAxis, 
  Tooltip, 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  CartesianGrid
} from "recharts"
import { 
  Brain, 
  Loader2, 
  Activity, 
  Compass, 
  Hash, 
  Layers, 
  FileText, 
  Flame,
  AlertCircle,
  Plus,
  Trash2,
  BookOpen,
  CheckCircle2,
  Upload,
  Settings,
  ChevronDown,
  PanelLeftClose,
  PanelLeftOpen,
  Sliders
} from "lucide-react"

const fixMergedWords = (text: string) => {
  if (!text) return text
  let fixed = text
    .replace(/\bTothebestoftheauthors\b/gi, "To the best of the authors")
    .replace(/\bthisisthefirstrecommendationsystem\b/gi, "this is the first recommendation system")
    .replace(/\bmeasuresPandQ\b/gi, "measures P and Q")
    .replace(/\bTothebestofourknowledge\b/gi, "To the best of our knowledge")
    .replace(/\bLEffective\b/g, "Effective")
    .replace(/\bTothebest\b/gi, "To the best")
    .replace(/\bcompetencyquestions\b/gi, "competency questions")
    .replace(/([a-z])([A-Z])/g, '$1 $2') // Split camelCase runs
    .replace(/,([a-zA-Z])/g, ', $1')    // Space after comma
    .replace(/\.([a-zA-Z])/g, '. $1')    // Space after period
  return fixed
}

export default function NLPWorkspace() {
  const { 
    documents, 
    setDocuments,
    addDocument,
    removeDocument,
    params, 
    setParams, 
    currentJob, 
    setCurrentJob, 
    pastJobs, 
    setPastJobs, 
    isLoading, 
    setLoading, 
    error, 
    setError 
  } = useNLPStore()

  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState("clustering")
  const [preloadKey, setPreloadKey] = useState("primary")
  const [isPreloading, setIsPreloading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isLeftPanelMinimized, setIsLeftPanelMinimized] = useState(false)
  
  // Modal State
  const [detailOpen, setDetailOpen] = useState(false)
  const [modalDocName, setModalDocName] = useState("")
  const [modalRawText, setModalRawText] = useState("")
  const [modalCleanedText, setModalCleanedText] = useState("")
  const [modalKeywords, setModalKeywords] = useState<string[]>([])
  const [modalSummaries, setModalSummaries] = useState<string[]>([])
  const [modalTab, setModalTab] = useState("highlighted")

  // Fetch initial documents & runs
  useEffect(() => {
    const initWorkspace = async () => {
      try {
        const docs = await api.documents.list()
        setDocuments(docs)
        const runs = await api.nlp.list()
        setPastJobs(runs)
      } catch (err: any) {
        console.error("Failed to fetch initial workspace state", err)
      }
    }
    initWorkspace()
  }, [setDocuments, setPastJobs])

  const handleSelectAllDocs = () => {
    if (selectedDocIds.length === documents.length) {
      setSelectedDocIds([])
    } else {
      setSelectedDocIds(documents.map((d) => d.id))
    }
  }

  const handleToggleDoc = (id: string) => {
    setSelectedDocIds((prev) => 
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    )
  }

  const handlePreload = async () => {
    setIsPreloading(true)
    setError(null)
    try {
      const docs = await api.documents.preload(preloadKey)
      setDocuments(docs)
      setSelectedDocIds(docs.map(d => d.id))
    } catch (err: any) {
      setError(err.message || "Failed to preload sample corpus.")
    } finally {
      setIsPreloading(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return
    setIsUploading(true)
    setError(null)
    try {
      const fileArray = Array.from(e.target.files)
      const uploaded = await api.documents.upload(fileArray)
      uploaded.forEach((doc) => addDocument(doc))
    } catch (err: any) {
      setError(err.message || "Failed to upload files.")
    } finally {
      setIsUploading(false)
    }
  }

  const handleDeleteDoc = async (id: string) => {
    setError(null)
    try {
      await api.documents.delete(id)
      removeDocument(id)
      setSelectedDocIds(prev => prev.filter(item => item !== id))
    } catch (err: any) {
      setError(err.message || "Failed to delete document.")
    }
  }

  const handleRunPipeline = async () => {
    if (selectedDocIds.length < 2) {
      setError("Please select at least 2 documents to run the analysis.")
      return
    }
    setLoading(true)
    setError(null)
    try {
      const results = await api.nlp.analyze(selectedDocIds, params)
      setCurrentJob(results)
      
      const jobs = await api.nlp.list()
      setPastJobs(jobs)
    } catch (err: any) {
      setError(err.message || "Failed to execute NLP analysis pipeline.")
    } finally {
      setLoading(false)
    }
  }

  const handleOpenDocModal = (docId: string, docName: string, clusterId: number) => {
    if (!currentJob) return
    const details = currentJob.results.document_details[docId]
    if (!details) return
    
    setModalDocName(docName)
    setModalRawText(details.raw_text)
    setModalCleanedText(details.cleaned_text)
    setModalKeywords(currentJob.results.clusters[clusterId]?.keywords || details.keywords)
    setModalSummaries(details.summary)
    setModalTab("highlighted")
    setDetailOpen(true)
  }

  // Text highlighter for modal
  const renderHighlightedText = () => {
    let text = modalCleanedText
    if (!text) return ""

    modalSummaries.forEach((sent) => {
      if (!sent.trim()) return
      const escaped = sent.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')
      const regex = new RegExp(`(${escaped})`, 'gi')
      text = text.replace(regex, `<mark class="summary-hl">$1</mark>`)
    })

    modalKeywords.forEach((word) => {
      const escaped = word.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')
      const regex = new RegExp(`\\b(${escaped})\\b`, 'gi')
      text = text.replace(regex, `<span class="keyword-hl">$1</span>`)
    })

    return (
      <div 
        className="leading-relaxed whitespace-pre-wrap text-zinc-700 dark:text-zinc-350 text-sm bg-zinc-50 dark:bg-zinc-950/65 p-4 border border-zinc-200 dark:border-zinc-800 rounded-lg max-h-[450px] overflow-y-auto" 
        dangerouslySetInnerHTML={{ __html: text }} 
      />
    )
  }

  const scatterData = currentJob?.results.pca_scatter || []
  
  const silhouetteChartData = currentJob?.results.scores_per_k 
    ? Object.entries(currentJob.results.scores_per_k).map(([k, score]) => ({
        k: parseInt(k),
        score: score >= -1 ? score : null
      }))
    : []

  const clusterColors = [
    "#3b82f6", // blue
    "#6366f1", // indigo
    "#10b981", // emerald
    "#f59e0b", // amber
    "#ec4899", // pink
    "#8b5cf6"  // violet
  ]

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 B"
    const k = 1024
    const sizes = ["B", "KB", "MB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i]
  }

  return (
    <div className="flex-1 flex flex-col max-w-6xl mx-auto w-full px-4 py-8 space-y-6 relative">
      
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* LEFT COLUMN: Workspace Controls & Config */}
        <div className={`${isLeftPanelMinimized ? "lg:col-span-1" : "lg:col-span-4"} space-y-6 transition-all duration-300`}>
          {isLeftPanelMinimized ? (
            <div className="flex flex-col items-center py-5 px-2 bg-white/70 dark:bg-zinc-950/45 border border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md rounded-2xl shadow-sm space-y-6 min-h-[400px] w-full animate-in fade-in-50 duration-300">
              <button
                onClick={() => setIsLeftPanelMinimized(false)}
                className="p-1.5 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900/60 transition-colors text-zinc-500 hover:text-zinc-900 dark:hover:text-white cursor-pointer border-0 bg-transparent"
                title="Expand Controls Panel"
              >
                <PanelLeftOpen className="w-4.5 h-4.5" />
              </button>
              <div className="border-b border-zinc-200 dark:border-zinc-800 w-full" />
              
              {/* Document Workspace minimized button */}
              <button
                onClick={() => setIsLeftPanelMinimized(false)}
                className="p-3 rounded-xl hover:bg-blue-500/10 dark:hover:bg-blue-500/20 text-zinc-400 hover:text-blue-500 transition-all cursor-pointer relative border-0 bg-transparent"
                title="Documents Workspace (Click to expand)"
              >
                <FileText className="w-5 h-5" />
                {documents.length > 0 && (
                  <span className="absolute top-1 right-1 w-4 h-4 rounded-full bg-blue-600 text-[8px] text-white flex items-center justify-center font-bold">
                    {selectedDocIds.length}
                  </span>
                )}
              </button>

              {/* Pipeline Parameters minimized button */}
              <button
                onClick={() => setIsLeftPanelMinimized(false)}
                className="p-3 rounded-xl hover:bg-indigo-500/10 dark:hover:bg-indigo-500/20 text-zinc-400 hover:text-indigo-500 transition-all cursor-pointer border-0 bg-transparent"
                title="Pipeline Parameters (Click to expand)"
              >
                <Sliders className="w-5 h-5" />
              </button>

              {/* Run Pipeline minimized button */}
              <button
                onClick={() => {
                  if (selectedDocIds.length >= 2 && !isLoading) {
                    handleRunPipeline();
                  } else {
                    setIsLeftPanelMinimized(false);
                  }
                }}
                disabled={isLoading}
                className={`p-3 rounded-xl transition-all cursor-pointer border-0 bg-transparent ${
                  selectedDocIds.length >= 2 
                    ? "hover:bg-emerald-500/10 dark:hover:bg-emerald-500/20 text-emerald-600 hover:text-emerald-500"
                    : "text-zinc-300 dark:text-zinc-800 cursor-not-allowed"
                }`}
                title={isLoading ? "Running..." : "Execute Pipeline (Click to run / expand)"}
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Activity className="w-5 h-5" />
                )}
              </button>
            </div>
          ) : (
            <>
              {/* Card 1: Document Workspace */}
              <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm flex flex-col h-[400px]">
                <CardHeader className="pb-3 border-b border-zinc-150 dark:border-zinc-900/60 flex flex-col shrink-0">
                  <div className="flex justify-between items-center">
                    <CardTitle className="text-sm font-bold text-zinc-900 dark:text-white m-0">Document Workspace</CardTitle>
                    <div className="flex items-center gap-2">
                      {documents.length > 0 && (
                        <button 
                          onClick={handleSelectAllDocs}
                          className="text-blue-600 dark:text-blue-400 text-xs font-bold hover:underline bg-transparent border-0 cursor-pointer"
                        >
                          {selectedDocIds.length === documents.length ? "Deselect All" : "Select All"}
                        </button>
                      )}
                      <button
                        onClick={() => setIsLeftPanelMinimized(true)}
                        className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white bg-transparent border-0 cursor-pointer p-0.5 rounded hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
                        title="Minimize Controls Panel"
                      >
                        <PanelLeftClose className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                  <CardDescription className="text-zinc-500 dark:text-zinc-450 text-xs mt-0.5">Upload or preload files to run mappings.</CardDescription>
                </CardHeader>
                
                {/* Scrollable list of docs */}
                <CardContent className="flex-1 overflow-y-auto p-3 space-y-2">
                  {documents.length > 0 ? (
                    documents.map((doc) => {
                      const isSelected = selectedDocIds.includes(doc.id)
                      return (
                        <div
                          key={doc.id}
                          onClick={() => handleToggleDoc(doc.id)}
                          className={`flex items-center justify-between p-2 rounded-lg border text-xs cursor-pointer transition-all ${
                            isSelected 
                              ? "border-blue-500/30 bg-blue-500/5 text-zinc-955 dark:text-white font-semibold shadow-xs" 
                              : "border-zinc-100 dark:border-zinc-900/50 hover:bg-zinc-50 dark:hover:bg-zinc-900/30 text-zinc-500 dark:text-zinc-400"
                          }`}
                        >
                          <div className="flex items-center gap-1.5 min-w-0">
                            <div className={`w-3.5 h-3.5 rounded border flex items-center justify-center shrink-0 transition-all ${
                              isSelected 
                                ? "bg-blue-600 border-blue-600 text-white" 
                                : "border-zinc-300 dark:border-zinc-800"
                            }`}>
                              {isSelected && <span className="text-[10px] font-bold">✓</span>}
                            </div>
                            <FileText className="w-3.5 h-3.5 shrink-0 text-zinc-400" />
                            <span className="truncate font-medium">{doc.filename}</span>
                          </div>
                          
                          <div className="flex items-center gap-1.5 shrink-0 ml-1">
                            <span className="text-xs text-zinc-400 font-mono">{formatBytes(doc.file_size)}</span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleDeleteDoc(doc.id)
                              }}
                              className="text-zinc-400 hover:text-destructive p-1 bg-transparent border-0 cursor-pointer rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-900/60 transition-colors"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      )
                    })
                  ) : (
                    <div className="text-center py-12 text-zinc-400 text-xs border border-dashed border-zinc-200 dark:border-zinc-800 rounded-xl bg-zinc-50/20 dark:bg-zinc-950/20">
                      No documents loaded.<br />Preload a sample corpus or upload custom files.
                    </div>
                  )}
                </CardContent>

                {/* Bottom Actions section */}
                <div className="p-3 border-t border-zinc-150 dark:border-zinc-900 bg-zinc-50/50 dark:bg-zinc-950/20 shrink-0 space-y-2.5">
                  
                  {/* Load Sample Selector */}
                  <div className="flex items-center gap-1.5 border border-zinc-200 dark:border-zinc-800 rounded-xl p-1 bg-white dark:bg-zinc-950 w-full shadow-xs">
                    <Select
                      value={preloadKey}
                      onChange={(e: any) => setPreloadKey(e.target.value)}
                      className="h-7 py-0.5 text-xs w-full bg-transparent border-0 text-zinc-800 dark:text-zinc-200 focus-visible:ring-0 focus:border-0 shadow-none"
                    >
                      <option value="primary">Primary Corpus</option>
                      <option value="pdf_papers">AI PDF Papers</option>
                      <option value="semantic_demo">Semantic Demo</option>
                    </Select>
                    <Button
                      size="sm"
                      onClick={handlePreload}
                      disabled={isPreloading}
                      className="h-7 px-3 bg-zinc-900 dark:bg-zinc-800 hover:bg-zinc-800 dark:hover:bg-zinc-700 text-white text-xs rounded-lg font-bold cursor-pointer shrink-0"
                    >
                      {isPreloading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <BookOpen className="w-3 h-3" />}
                      <span className="ml-1">Load</span>
                    </Button>
                  </div>

                  {/* Upload Custom Files */}
                  <label className="relative block w-full">
                    <span className="h-8.5 w-full bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-900 dark:hover:bg-zinc-800 text-zinc-850 dark:text-zinc-200 border border-zinc-200 dark:border-zinc-800 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer transition-colors shadow-xs">
                      {isUploading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5 text-zinc-400" />}
                      <span>Upload Custom Files</span>
                    </span>
                    <input
                      type="file"
                      multiple
                      accept=".txt,.pdf"
                      onChange={handleFileUpload}
                      className="hidden"
                      disabled={isUploading || isPreloading}
                    />
                  </label>
                </div>
              </Card>

              {/* Card 2: Pipeline Parameters */}
              <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm p-5 flex flex-col justify-between space-y-4">
                <div className="space-y-4">
                  <div>
                    <CardTitle className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-1.5 m-0">
                      <Settings className="w-3.5 h-3.5 text-blue-500" />
                      Pipeline Parameters
                    </CardTitle>
                    <CardDescription className="text-zinc-550 dark:text-zinc-455 text-xs mt-0.5">Customize feature vectorization extraction settings.</CardDescription>
                  </div>

                  <div className="space-y-3.5">
                    {/* Mode Select */}
                    <div className="space-y-1">
                      <label className="text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider block">Vectorization Mode</label>
                      <Select
                        value={params.vectorization_mode}
                        onChange={(e: any) => setParams({ vectorization_mode: e.target.value })}
                        className="h-8 py-0.5 text-xs border-zinc-200 dark:border-zinc-850 bg-white dark:bg-zinc-950 text-zinc-850 dark:text-zinc-200"
                      >
                        <option value="TF-IDF (Classical)">TF-IDF (Classical)</option>
                        <option value="Semantic Embeddings (SBERT)">Semantic (SBERT)</option>
                      </Select>
                    </div>

                    {/* Numbers filter */}
                    <div className="flex items-center justify-between p-2 rounded-lg bg-zinc-50 dark:bg-zinc-900/30 border border-zinc-150 dark:border-zinc-900 h-8">
                      <span className="text-xs font-semibold text-zinc-650 dark:text-zinc-400">Retain Numeric Stats</span>
                      <Switch
                        checked={params.preserve_numbers}
                        disabled={!params.vectorization_mode.includes("TF-IDF")}
                        onCheckedChange={(val) => setParams({ preserve_numbers: val })}
                        className="scale-75"
                      />
                    </div>

                    {/* Slider 1: clusters */}
                    <div className="space-y-1">
                      <div className="flex justify-between items-center text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider">
                        <span>Clusters (k):</span>
                        <span className="text-blue-600 dark:text-blue-400 font-bold">{params.k_clusters}</span>
                      </div>
                      <input
                        type="range"
                        min="2"
                        max={Math.max(2, documents.length)}
                        value={params.k_clusters}
                        onChange={(e) => setParams({ k_clusters: parseInt(e.target.value) })}
                        className="w-full h-1 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-blue-600 my-1"
                      />
                    </div>

                    {/* Slider 2: LDA topics */}
                    <div className="space-y-1">
                      <div className="flex justify-between items-center text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider">
                        <span>LDA Topics:</span>
                        <span className="text-indigo-600 dark:text-indigo-400 font-bold">{params.n_topics}</span>
                      </div>
                      <input
                        type="range"
                        min="2"
                        max={Math.max(2, Math.min(6, documents.length))}
                        value={params.n_topics}
                        onChange={(e) => setParams({ n_topics: parseInt(e.target.value) })}
                        className="w-full h-1 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-indigo-600 my-1"
                      />
                    </div>
                  </div>
                </div>

                {/* Run button */}
                <div className="border-t border-zinc-150 dark:border-zinc-900 pt-3.5 shrink-0">
                  <Button
                    onClick={handleRunPipeline}
                    disabled={isLoading || selectedDocIds.length < 2}
                    className="w-full font-semibold bg-gradient-to-r from-blue-600 to-indigo-500 hover:from-blue-500 hover:to-indigo-400 text-white shadow-md shadow-blue-500/10 cursor-pointer text-xs h-9.5 rounded-xl transition-all active:scale-[0.98]"
                  >
                    {isLoading ? (
                      <span className="flex items-center gap-1.5 justify-center">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        Running Pipeline...
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 justify-center">
                        <Activity className="w-3.5 h-3.5" />
                        Execute Analysis
                      </span>
                    )}
                  </Button>
                </div>
              </Card>
            </>
          )}
        </div>

        {/* RIGHT COLUMN: Results Dashboard */}
        <div className={`${isLeftPanelMinimized ? "lg:col-span-11" : "lg:col-span-8"} flex flex-col space-y-6 transition-all duration-300`}>
          {isLoading ? (
            <div className="flex flex-col items-center justify-center border border-zinc-200 dark:border-zinc-800 border-dashed rounded-2xl p-16 text-center min-h-[400px] relative overflow-hidden bg-white/30 dark:bg-zinc-950/20 backdrop-blur-md animate-pulse">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-blue-500/10 dark:bg-blue-600/10 rounded-full blur-[40px] pointer-events-none -z-10" />

              <Loader2 className="w-12 h-12 text-blue-600 dark:text-blue-400 mb-5 animate-spin" />
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white m-0">Analyzing Document Corpus...</h3>
              <p className="text-xs text-zinc-550 dark:text-zinc-450 mt-2.5 max-w-sm leading-relaxed font-light">
                Executing text preprocessing, feature vectorization, cluster mapping, and similarity computation. This will take just a moment.
              </p>
            </div>
          ) : currentJob ? (
            <div className="w-full space-y-4">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
                
                {/* Pill tabs list */}
                <div className="flex justify-between items-center border-b border-zinc-200 dark:border-zinc-800 pb-2">
                  <TabsList className="bg-zinc-100 dark:bg-zinc-900/60 p-0.5 border border-zinc-200 dark:border-zinc-800/80 rounded-full">
                    <TabsTrigger value="clustering" className="text-sm px-4 py-1.5 gap-1.5 data-[state=active]:bg-white dark:data-[state=active]:bg-zinc-950 rounded-full font-bold">
                      <Layers className="w-3.5 h-3.5 text-blue-500" />
                      Semantic Clusters
                    </TabsTrigger>
                    <TabsTrigger value="lda" className="text-sm px-4 py-1.5 gap-1.5 data-[state=active]:bg-white dark:data-[state=active]:bg-zinc-950 rounded-full font-bold">
                      <Hash className="w-3.5 h-3.5 text-indigo-550" />
                      LDA Themes
                    </TabsTrigger>
                    <TabsTrigger value="similarity" className="text-sm px-4 py-1.5 gap-1.5 data-[state=active]:bg-white dark:data-[state=active]:bg-zinc-950 rounded-full font-bold">
                      <Flame className="w-3.5 h-3.5 text-rose-550" />
                      Similarity Heatmap
                    </TabsTrigger>
                  </TabsList>
                  
                  <span className="text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider">
                    Run Vectorizer: <span className="text-blue-600 dark:text-blue-400">{currentJob.results.vectorization_mode}</span>
                  </span>
                </div>

                {/* Tab Content: Clustering (Charts side-by-side) */}
                <TabsContent value="clustering" className="space-y-6 outline-none">
                  
                  {/* Two charts side-by-side using full width */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    
                    {/* 2D PCA Scatter plot */}
                    <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-1.5 m-0">
                          <Compass className="w-3.5 h-3.5 text-blue-500" />
                          PCA Projection (2D Clusters)
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="h-[250px] pt-2">
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-zinc-100 dark:text-zinc-900" />
                            <XAxis type="number" dataKey="x" name="PCA 1" stroke="#a1a1aa" fontSize={10} />
                            <YAxis type="number" dataKey="y" name="PCA 2" stroke="#a1a1aa" fontSize={10} />
                            <ZAxis type="category" dataKey="name" name="Document" />
                            <Tooltip 
                              cursor={{ strokeDasharray: '3 3' }} 
                              content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                  const data = payload[0].payload as PCAPoint
                                  const clr = clusterColors[data.cluster % clusterColors.length]
                                  return (
                                    <div 
                                      className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 p-2.5 rounded-lg shadow-xl text-xs"
                                      style={{ borderLeft: `3px solid ${clr}` }}
                                    >
                                      <p className="font-bold text-zinc-900 dark:text-white">{data.name}</p>
                                      <p className="text-zinc-550 mt-0.5">Cluster ID: <span className="font-bold" style={{ color: clr }}>{data.cluster}</span></p>
                                    </div>
                                  )
                                }
                                return null
                              }}
                            />
                            {Array.from(new Set(scatterData.map(d => d.cluster))).map((clusterId) => (
                              <Scatter
                                key={clusterId}
                                name={`Cluster ${clusterId}`}
                                data={scatterData.filter(d => d.cluster === clusterId)}
                                fill={clusterColors[clusterId % clusterColors.length]}
                              />
                            ))}
                          </ScatterChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>

                    {/* Silhouette index */}
                    <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-1.5 m-0">
                          <Activity className="w-3.5 h-3.5 text-blue-500" />
                          Silhouette Index Optimization
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="h-[250px] pt-2">
                        {silhouetteChartData.length > 0 ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={silhouetteChartData} margin={{ top: 10, right: 10, bottom: 10, left: -25 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-zinc-100 dark:text-zinc-900" />
                              <XAxis dataKey="k" stroke="#a1a1aa" fontSize={10} />
                              <YAxis stroke="#a1a1aa" fontSize={10} domain={['auto', 'auto']} />
                              <Tooltip
                                content={({ active, payload }) => {
                                  if (active && payload && payload.length) {
                                    return (
                                      <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 p-2 rounded-lg shadow-xl text-xs border-l-3 border-blue-500">
                                        <p className="font-semibold text-zinc-900 dark:text-white">Clusters (k): {payload[0].payload.k}</p>
                                        <p className="text-blue-500 mt-0.5 font-bold">Silhouette Score: {payload[0].payload.score?.toFixed(4)}</p>
                                      </div>
                                    )
                                  }
                                  return null
                                }}
                              />
                              <Line
                                type="monotone"
                                dataKey="score"
                                stroke="#3b82f6"
                                strokeWidth={2}
                                activeDot={{ r: 5 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="flex items-center justify-center h-full text-zinc-400 text-xs text-center px-4 font-light">
                            Requires 3+ documents to compute silhouette optimization scores.
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>                  {/* Cluster Themes Grid (Full width cards below) */}
                  <div className="space-y-4 pt-2">
                    <span className="text-xs font-bold text-zinc-400 dark:text-zinc-550 uppercase tracking-wider block">Discovered Cluster Themes</span>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {currentJob.results.clusters.map((cluster) => (
                        <Card key={cluster.cluster_id} className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm animate-fade-in">
                          <CardHeader className="p-4 flex flex-row justify-between items-center border-b border-zinc-150 dark:border-zinc-900/60">
                            <span 
                              className="text-xs font-bold text-white px-2.5 py-0.5 rounded-full uppercase tracking-wider shadow-xs"
                              style={{ backgroundColor: clusterColors[cluster.cluster_id % clusterColors.length] }}
                            >
                              Cluster {cluster.cluster_id}
                            </span>
                            <span className="text-xs text-zinc-400 dark:text-zinc-550 font-semibold">{cluster.documents.length} files</span>
                          </CardHeader>
                          <CardContent className="p-4 space-y-4">
                            
                            {/* Vocabulary */}
                            <div className="space-y-1">
                              <span className="text-xs font-semibold text-zinc-450 dark:text-zinc-500 uppercase tracking-wider block">Core Vocabulary</span>
                              <div className="flex flex-wrap gap-1">
                                {cluster.keywords.map((kw, idx) => (
                                  <span key={idx} className="px-2 py-0.5 rounded text-xs bg-zinc-100 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 border border-zinc-200/50 dark:border-zinc-800/80">
                                    {kw}
                                  </span>
                                ))}
                              </div>
                            </div>

                            {/* Extractive Summary */}
                            <div className="space-y-2 pt-2 border-t border-zinc-150 dark:border-zinc-900">
                              <span className="text-xs font-semibold text-zinc-455 dark:text-zinc-500 uppercase tracking-wider block">Extractive Highlights</span>
                              <div className="space-y-2 max-h-[160px] overflow-y-auto pr-1">
                                {cluster.documents.map((d) => (
                                  <div key={d.id} className="p-2.5 rounded bg-zinc-50/50 dark:bg-zinc-950/60 border border-zinc-150 dark:border-zinc-900/50 text-sm leading-relaxed text-zinc-650 dark:text-zinc-400 font-light">
                                    <button
                                      onClick={() => handleOpenDocModal(d.id, d.filename, cluster.cluster_id)}
                                      className="font-semibold text-blue-600 dark:text-blue-400 hover:underline bg-transparent border-0 p-0 text-left cursor-pointer block mb-1 truncate w-full text-sm"
                                    >
                                      {d.filename}
                                    </button>
                                    {d.summary && d.summary.length > 0 ? fixMergedWords(d.summary.join(" ")) : "No summary highlights."}
                                  </div>
                                ))}
                              </div>
                            </div>

                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>

                </TabsContent>

                {/* Tab Content: LDA Topics */}
                <TabsContent value="lda" className="outline-none">
                  <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                    <CardHeader className="pb-3 border-b border-zinc-150 dark:border-zinc-900">
                      <CardTitle className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-1.5 m-0">
                        <Hash className="w-4 h-4 text-indigo-550" />
                        Theme Matrix (LDA Topic Modeling)
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4 space-y-3">
                      {currentJob.results.topics && currentJob.results.topics.length > 0 ? (
                        currentJob.results.topics.map((topic, idx) => (
                          <div key={idx} className="flex flex-col sm:flex-row sm:items-center p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-900/20 border border-zinc-150 dark:border-zinc-850/50 gap-4">
                            <div className="flex items-center gap-2 w-28 shrink-0">
                              <div className="w-6.5 h-6.5 rounded bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-xs font-bold text-indigo-650 dark:text-indigo-400">
                                T{idx+1}
                              </div>
                              <span className="font-bold text-xs text-zinc-800 dark:text-white">Theme {idx+1}</span>
                            </div>
                            
                            <div className="flex flex-wrap gap-1">
                              {topic.split(", ").map((kw, i) => (
                                <span key={i} className="px-2.5 py-0.5 rounded text-xs bg-white dark:bg-zinc-950 text-zinc-650 dark:text-zinc-350 border border-zinc-200 dark:border-zinc-800/85">
                                  {kw}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-10 text-zinc-400 text-xs font-light">
                          Topic modeling is not applicable for single documents.
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Tab Content: Similarity Heatmap (Full width centered) */}
                <TabsContent value="similarity" className="outline-none">
                  <Card className="bg-white/70 dark:bg-zinc-950/45 border-zinc-200 dark:border-zinc-800/80 backdrop-blur-md shadow-sm">
                    <CardHeader className="pb-3 border-b border-zinc-150 dark:border-zinc-900">
                      <CardTitle className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-1.5 m-0">
                        <Flame className="w-4 h-4 text-rose-550" />
                        Document Cosine Similarity Heatmap
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-6 overflow-x-auto">
                      {currentJob.results.similarity && currentJob.results.similarity.matrix.length >= 2 ? (
                        <div className="w-full flex justify-center py-4">
                          <table className="border-collapse">
                            <thead>
                              <tr>
                                <th className="p-2"></th>
                                {currentJob.results.similarity.filenames.map((name, i) => (
                                  <th key={i} className="p-2 text-xs font-bold text-zinc-400 text-center truncate max-w-[100px] border-b border-zinc-200 dark:border-zinc-800" title={name}>
                                    {name.substring(0, 10)}{name.length > 10 ? "..." : ""}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {currentJob.results.similarity.matrix.map((row, rIdx) => (
                                <tr key={rIdx}>
                                  <td className="p-2 text-xs font-bold text-zinc-400 text-right truncate max-w-[100px] border-r border-zinc-200 dark:border-zinc-800 pr-3" title={currentJob.results.similarity.filenames[rIdx]}>
                                    {currentJob.results.similarity.filenames[rIdx].substring(0, 10)}{currentJob.results.similarity.filenames[rIdx].length > 10 ? "..." : ""}
                                  </td>
                                  {row.map((val, cIdx) => {
                                    const cellBg = `rgba(59, 130, 246, ${val * 0.88})`
                                    return (
                                      <td 
                                        key={cIdx} 
                                        style={{ backgroundColor: cellBg }}
                                        className={`w-14 h-14 text-center text-xs font-bold font-mono border border-white dark:border-zinc-950 transition-all hover:scale-105 relative group cursor-help shadow-xs ${val > 0.55 ? 'text-white' : 'text-zinc-800 dark:text-zinc-200'}`}
                                      >
                                        {val.toFixed(2)}
                                        <div className="absolute hidden group-hover:block bottom-14 left-1/2 -translate-x-1/2 z-50 bg-zinc-950 text-white text-xs p-2.5 rounded border border-zinc-800 shadow-xl whitespace-nowrap">
                                          <p className="font-bold">{val.toFixed(4)}</p>
                                          <p className="text-zinc-550 mt-0.5">{currentJob.results.similarity.filenames[rIdx]} &harr; {currentJob.results.similarity.filenames[cIdx]}</p>
                                        </div>
                                      </td>
                                    )
                                  })}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="text-center py-10 text-zinc-400 text-xs font-light">
                          Need at least 2 documents to construct similarity heatmaps.
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

              </Tabs>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center border border-zinc-200 dark:border-zinc-800 border-dashed rounded-2xl p-16 text-center min-h-[400px] relative overflow-hidden bg-white/30 dark:bg-zinc-950/20 backdrop-blur-md">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-blue-500/5 dark:bg-blue-600/5 rounded-full blur-[40px] pointer-events-none -z-10" />

              <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-600 dark:text-blue-400 mb-5 border border-blue-500/5 animate-pulse">
                <Brain className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white m-0">No Active Analysis Run</h3>
              <p className="text-xs text-zinc-550 dark:text-zinc-450 mt-2.5 max-w-sm leading-relaxed font-light">
                Select two or more documents from the checklist on the left, set your parameters, and click "Execute Analysis" to generate similarity matrices and cluster models.
              </p>
            </div>
          )}
        </div>

      </div>

      {/* Document Detail Overlay Dialog Modal */}
      <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
        <DialogContent className="max-w-3xl border-zinc-250 dark:border-zinc-800 bg-white/95 dark:bg-zinc-950/95 max-h-[90vh] flex flex-col p-6 shadow-2xl">
          <DialogHeader className="pb-2">
            <DialogTitle className="text-lg font-bold text-zinc-900 dark:text-white truncate m-0">{modalDocName}</DialogTitle>
            <DialogDescription className="text-zinc-550 flex items-center gap-3 mt-1.5 text-xs font-semibold">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-green-500/20 border border-green-500 inline-block"></span>
                Extractive Summary Highlights
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-red-500/20 border-b border-red-500 inline-block"></span>
                Theme Keywords
              </span>
            </DialogDescription>
          </DialogHeader>
          
          <Tabs value={modalTab} onValueChange={setModalTab} className="flex-1 overflow-hidden flex flex-col pt-2">
            <TabsList className="bg-zinc-100 dark:bg-zinc-900 p-0.5 border border-zinc-200 dark:border-zinc-800 self-start">
              <TabsTrigger value="highlighted" className="text-xs px-3 py-1 data-[state=active]:bg-white dark:data-[state=active]:bg-zinc-950">Highlighted Highlights</TabsTrigger>
              <TabsTrigger value="original" className="text-xs px-3 py-1 data-[state=active]:bg-white dark:data-[state=active]:bg-zinc-950">Original Text</TabsTrigger>
            </TabsList>
            
            <TabsContent value="highlighted" className="flex-1 overflow-y-auto mt-4 outline-none">
              {renderHighlightedText()}
            </TabsContent>
            
            <TabsContent value="original" className="flex-1 overflow-y-auto mt-4 outline-none">
              <div className="leading-relaxed text-zinc-650 dark:text-zinc-455 text-sm bg-zinc-50 dark:bg-zinc-950/60 p-4 border border-zinc-200 dark:border-zinc-800 rounded-lg max-h-[450px] overflow-y-auto font-sans whitespace-pre-wrap">
                {modalRawText}
              </div>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
    </div>
  )
}
