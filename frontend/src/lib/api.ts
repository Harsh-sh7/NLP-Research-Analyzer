const API_BASE = "http://localhost:8000/api"

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
  
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers
  })
  
  if (response.status === 204) {
    return null as unknown as T
  }
  
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.detail || "Something went wrong")
  }
  
  return data as T
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
