import React, { useEffect, useState } from "react"
import { useResearchStore, Report } from "@/store/researchStore"
import { api } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { 
  FileText, 
  Download, 
  Trash2, 
  Link, 
  Clock, 
  BookOpen, 
  Sparkles,
  AlertCircle
} from "lucide-react"

export default function ReportsExplorer() {
  const { reports, setReports } = useResearchStore()
  const [selectedReport, setSelectedReport] = useState<Report | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    const loadReports = async () => {
      try {
        const list = await api.research.listReports()
        setReports(list)
        if (list.length > 0) {
          setSelectedReport(list[0])
        }
      } catch (err) {
        console.error("Failed to load reports", err)
      }
    }
    loadReports()
  }, [setReports])

  const handleDeleteReport = async (jobId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setErrorMsg(null)
    try {
      await api.research.delete(jobId)
      const list = reports.filter((r) => r.job_id !== jobId)
      setReports(list)
      if (selectedReport?.job_id === jobId) {
        setSelectedReport(list.length > 0 ? list[0] : null)
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to delete report.")
    }
  }

  const handleDownload = (report: Report) => {
    const blob = new Blob([report.content], { type: "text/markdown" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${report.title.toLowerCase().replace(/[^a-z0-9]/g, "_")}_report.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const formatReportText = (text: string) => {
    if (!text) return ""
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      
    const citationRegex = /\[Source:\s*([^\]]+)\]/g
    html = html.replace(citationRegex, (match, url) => {
      return `<a href="${url}" target="_blank" rel="noreferrer" class="inline-flex items-center gap-0.5 px-2 py-0.5 rounded bg-primary/10 border border-primary/20 text-xs font-bold text-primary hover:underline hover:bg-primary/20 transition-all ml-1"><span class="w-1.5 h-1.5 bg-primary rounded-full"></span>Cite</a>`
    })

    html = html
      .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-extrabold text-white mt-6 mb-3 border-b border-white/5 pb-1">$1</h1>')
      .replace(/^## (.*$)/gim, '<h2 class="text-lg font-bold text-white mt-5 mb-2.5">$1</h2>')
      .replace(/^### (.*$)/gim, '<h3 class="text-sm font-bold text-zinc-200 mt-4 mb-2">$1</h3>')
      .replace(/^\* (.*$)/gim, '<li class="ml-4 list-disc text-zinc-350 py-0.5 text-sm">$1</li>')
      .replace(/^- (.*$)/gim, '<li class="ml-4 list-disc text-zinc-350 py-0.5 text-sm">$1</li>')
      .replace(/\n\n/g, '<p class="my-2.5 text-zinc-350 leading-relaxed text-sm font-light"></p>')

    return <div className="space-y-1" dangerouslySetInnerHTML={{ __html: html }} />
  }

  return (
    <div className="space-y-8 p-6 max-w-7xl mx-auto min-h-[calc(100vh-80px)]">
      {/* Top Banner */}
      <div className="border-b border-white/5 pb-6">
        <h1 className="text-3xl font-bold tracking-tight text-white m-0 leading-tight">Reports Repository</h1>
        <p className="text-zinc-400 text-sm mt-1">Browse, view, and export all generated synthesis research documents.</p>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm flex items-start gap-2">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {reports.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Reports List sidebar (1/3 width) */}
          <div className="lg:col-span-1 space-y-4">
            <h3 className="text-sm font-bold text-zinc-550 uppercase tracking-wider block">Generated Reports</h3>
            <div className="space-y-3 max-h-[calc(100vh-250px)] overflow-y-auto pr-1">
              {reports.map((report) => (
                <div
                  key={report.id}
                  onClick={() => setSelectedReport(report)}
                  className={`p-4 rounded-xl border text-left cursor-pointer transition-all duration-200 ${
                    selectedReport?.id === report.id
                      ? "bg-primary/5 border-primary/25 shadow-md shadow-primary/5"
                      : "bg-zinc-950/40 border-zinc-850 text-zinc-400 hover:border-zinc-700"
                  }`}
                >
                  <div className="flex justify-between items-start gap-2">
                    <h4 className="text-sm font-semibold text-white truncate flex-1 m-0">{report.title}</h4>
                    <button
                      onClick={(e) => handleDeleteReport(report.job_id, e)}
                      className="text-zinc-550 hover:text-destructive transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <p className="text-xs text-zinc-550 mt-2 truncate font-light">Query: {report.title}</p>
                  
                  <div className="flex justify-between items-center mt-3 pt-2.5 border-t border-white/5 text-xs text-zinc-550">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {new Date(report.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                    </span>
                    <span className="flex items-center gap-1 font-semibold text-primary">
                      <BookOpen className="w-3 h-3" />
                      {report.metrics.sources} sources
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Reader Panel (2/3 width) */}
          <div className="lg:col-span-2">
            {selectedReport ? (
              <Card className="bg-zinc-950/40 border-zinc-800 flex flex-col h-full">
                <CardHeader className="border-b border-white/5 pb-4 flex flex-row justify-between items-center">
                  <div>
                    <CardTitle className="text-lg font-bold text-white m-0 truncate max-w-md">{selectedReport.title}</CardTitle>
                    <CardDescription className="text-zinc-500 mt-1 flex items-center gap-2">
                      <Clock className="w-3 h-3" />
                      Created {new Date(selectedReport.created_at).toLocaleString()}
                    </CardDescription>
                  </div>
                  
                  <Button
                    onClick={() => handleDownload(selectedReport)}
                    className="bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-white gap-1.5 cursor-pointer text-xs font-semibold py-1.5 px-3"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download MD
                  </Button>
                </CardHeader>
                
                <CardContent className="p-8 overflow-y-auto max-h-[calc(100vh-320px)] flex-1">
                  {/* Markdown content */}
                  <div className="prose prose-invert max-w-none">
                    {formatReportText(selectedReport.content)}
                  </div>
                  
                  {/* Sources index */}
                  {selectedReport.citations && Object.keys(selectedReport.citations).length > 0 && (
                    <div className="mt-8 pt-6 border-t border-white/5">
                      <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-1.5">
                        <Link className="w-4 h-4 text-primary" />
                        Reference Library
                      </h4>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {Array.from(new Set(Object.values(selectedReport.citations))).map((url: any, idx) => (
                          <a
                            key={idx}
                            href={url}
                            target="_blank"
                            rel="noreferrer"
                            className="p-2.5 rounded-lg bg-zinc-950/60 border border-zinc-850 text-xs text-zinc-400 hover:text-white hover:border-zinc-700 truncate block transition-all leading-normal"
                          >
                            {url}
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ) : null}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center border border-zinc-850 border-dashed rounded-2xl p-16 text-center max-w-xl mx-auto mt-12">
          <div className="w-12 h-12 rounded-full bg-secondary/10 flex items-center justify-center text-secondary mb-4">
            <Sparkles className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-white m-0">No Reports Available</h3>
          <p className="text-xs text-zinc-500 mt-2 max-w-xs leading-relaxed font-light">
            You haven't generated any research reports yet. Launch a research agent in the workspace to synthesize your first report.
          </p>
        </div>
      )}
    </div>
  )
}
