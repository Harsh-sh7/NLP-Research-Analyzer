import { useUIStore } from "@/store/uiStore"

const getApiBase = () => {
  let url = (import.meta.env.VITE_API_URL || "http://localhost:8000/api").trim().replace(/\/+$/, "")
  if (!url.endsWith("/api")) {
    url += "/api"
  }
  return url
}
export const API_BASE = getApiBase()

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem("token")
  
  const headers = new Headers(options.headers || {})
  if (token) {
    headers.set("Authorization", `Bearer ${token}`)
  }
  
  // Don't set Content-Type if uploading files (FormData handles its own boundaries)
  if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers
    })
    
    // Any successful HTTP response means the backend is online!
    useUIStore.getState().setBackendStatus("online")
    
    if (response.status === 204) {
      return null as unknown as T
    }
    
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.detail || "Something went wrong")
    }
    
    return data as T
  } catch (err: any) {
    // If it's a network error (like CORS blocked or server offline), set status to error
    if (err instanceof TypeError || (err.message && err.message.toLowerCase().includes("failed to fetch"))) {
      useUIStore.getState().setBackendStatus("error")
    }
    throw err
  }
}


export const api = {
  auth: {
    signup: (email: string, password: string, username?: string) => 
      request<any>("/auth/signup", {
        method: "POST",
        body: JSON.stringify({ email, password, username })
      }),
    login: (username: string, password: string) => {
      const form = new FormData()
      form.append("username", username)
      form.append("password", password)
      return request<{ access_token: string, token_type: string }>("/auth/login", {
        method: "POST",
        body: form
      })
    },
    getMe: () => request<any>("/auth/me")
  },
  
  documents: {
    list: () => request<any[]>("/documents/"),
    upload: (files: File[]) => {
      const form = new FormData()
      files.forEach((f) => form.append("files", f))
      return request<any[]>("/documents/upload", {
        method: "POST",
        body: form
      })
    },
    delete: (id: string) => 
      request<void>(`/documents/${id}`, {
        method: "DELETE"
      }),
    preload: (corpusKey: string) => 
      request<any[]>(`/documents/preload-corpus?corpus_key=${corpusKey}`, {
        method: "POST"
      })
  },
  
  nlp: {
    analyze: (documentIds: string[], params: any) => 
      request<any>("/nlp/analyze", {
        method: "POST",
        body: JSON.stringify({ document_ids: documentIds, parameters: params })
      }),
    list: () => request<any[]>("/nlp/jobs"),
    get: (id: string) => request<any>(`/nlp/jobs/${id}`)
  },
  
  research: {
    create: (query: string) => 
      request<any>("/research/", {
        method: "POST",
        body: JSON.stringify({ query })
      }),
    list: () => request<any[]>("/research/jobs"),
    get: (id: string) => request<any>(`/research/jobs/${id}`),
    delete: (id: string) => 
      request<void>(`/research/jobs/${id}`, {
        method: "DELETE"
      }),
    listReports: () => request<any[]>("/research/reports"),
    getReport: (id: string) => request<any>(`/research/reports/${id}`)
  }
}
