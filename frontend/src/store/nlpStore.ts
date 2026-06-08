import { create } from "zustand"

export interface NLPParams {
  vectorization_mode: string
  k_clusters: number
  preserve_numbers: boolean
  n_topics: number
}

export interface Document {
  id: string
  filename: string
  file_size: number
  created_at: string
}

export interface PCAPoint {
  id: string
  name: string
  x: number
  y: number
  cluster: number
}

export interface SimilarityData {
  matrix: number[][]
  filenames: string[]
  ids: string[]
}

export interface ClusterDoc {
  id: string
  filename: string
  summary: string[]
  keywords: string[]
}

export interface ClusterData {
  cluster_id: number
  keywords: string[]
  documents: ClusterDoc[]
}

export interface DocumentAnalysisDetail {
  summary: string[]
  keywords: string[]
  cleaned_text: string
  raw_text: string
}

export interface AnalysisResults {
  vectorization_mode: string
  k_clusters: number
  suggested_k: number
  scores_per_k: Record<number, number>
  pca_scatter: PCAPoint[]
  similarity: SimilarityData
  topics: string[]
  clusters: ClusterData[]
  document_details: Record<string, DocumentAnalysisDetail>
}

export interface AnalysisJob {
  id: string
  document_ids: string[]
  vectorization_mode: string
  parameters: NLPParams
  results: AnalysisResults
  created_at: string
}

interface NLPState {
  documents: Document[]
  params: NLPParams
  currentJob: AnalysisJob | null
  pastJobs: AnalysisJob[]
  isLoading: boolean
  error: string | null
  setDocuments: (docs: Document[]) => void
  addDocument: (doc: Document) => void
  removeDocument: (id: string) => void
  setParams: (params: Partial<NLPParams>) => void
  setCurrentJob: (job: AnalysisJob | null) => void
  setPastJobs: (jobs: AnalysisJob[]) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useNLPStore = create<NLPState>((set) => ({
  documents: [],
  params: {
    vectorization_mode: "TF-IDF (Classical)",
    k_clusters: 3,
    preserve_numbers: true,
    n_topics: 3,
  },
  currentJob: null,
  pastJobs: [],
  isLoading: false,
  error: null,
  setDocuments: (documents) => set({ documents }),
  addDocument: (doc) => set((state) => ({ documents: [...state.documents, doc] })),
  removeDocument: (id) => set((state) => ({ documents: state.documents.filter((d) => d.id !== id) })),
  setParams: (newParams) => set((state) => ({ params: { ...state.params, ...newParams } })),
  setCurrentJob: (currentJob) => set({ currentJob }),
  setPastJobs: (pastJobs) => set({ pastJobs }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}))
