import { create } from "zustand"

export interface Report {
  id: string
  job_id: string
  title: string
  content: string
  metrics: {
    tasks: number
    sources: number
    pages_scraped: number
    revisions: number
    context_chunks: number
  }
  citations: Record<string, string>
  created_at: string
}

export interface ResearchJob {
  id: string
  query: string
  status: string
  task_list: string[]
  scraped_urls: string[]
  citations: Record<string, string>
  report_draft: string
  revision_count: number
  created_at: string
  report?: Report
}

interface ResearchState {
  currentJob: ResearchJob | null
  activeNode: string | null // "planner" | "researcher" | "scraper" | "synthesizer" | "reviewer" | null
  logs: string[]
  taskProgress: {
    completed: string[]
    pending: string[]
  }
  feedback: string | null
  reports: Report[]
  jobs: ResearchJob[]
  isLoading: boolean
  error: string | null
  
  setCurrentJob: (job: ResearchJob | null) => void
  setActiveNode: (node: string | null) => void
  addLog: (log: string) => void
  clearLogs: () => void
  setTaskProgress: (completed: string[], pending: string[]) => void
  setFeedback: (feedback: string | null) => void
  setReports: (reports: Report[]) => void
  setJobs: (jobs: ResearchJob[]) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  resetActiveJob: () => void
}

export const useResearchStore = create<ResearchState>((set) => ({
  currentJob: null,
  activeNode: null,
  logs: [],
  taskProgress: { completed: [], pending: [] },
  feedback: null,
  reports: [],
  jobs: [],
  isLoading: false,
  error: null,
  
  setCurrentJob: (currentJob) => set({ currentJob }),
  setActiveNode: (activeNode) => set({ activeNode }),
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  clearLogs: () => set({ logs: [] }),
  setTaskProgress: (completed, pending) => set({ taskProgress: { completed, pending } }),
  setFeedback: (feedback) => set({ feedback }),
  setReports: (reports) => set({ reports }),
  setJobs: (jobs) => set({ jobs }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  resetActiveJob: () => set({
    currentJob: null,
    activeNode: null,
    logs: [],
    taskProgress: { completed: [], pending: [] },
    feedback: null,
    error: null,
  })
}))
