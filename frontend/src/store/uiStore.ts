import { create } from "zustand"

type PageType = "landing" | "auth" | "dashboard" | "nlp" | "research" | "reports" | "history" | "settings"
type ThemeType = "light" | "dark"

interface UIState {
  currentPage: PageType
  theme: ThemeType
  sidebarCollapsed: boolean
  activeDocId: string | null
  activeDocAnalysis: any | null
  backendStatus: "checking" | "online" | "sleeping" | "error"
  
  setPage: (page: PageType) => void
  toggleTheme: () => void
  toggleSidebar: () => void
  setSidebarCollapsed: (collapsed: boolean) => void
  openDocAnalysis: (docId: string, analysis: any) => void
  closeDocAnalysis: () => void
  setBackendStatus: (status: "checking" | "online" | "sleeping" | "error") => void
}

export const useUIStore = create<UIState>((set) => ({
  currentPage: "landing",
  theme: (localStorage.getItem("theme") as ThemeType) || "light",
  sidebarCollapsed: false,
  activeDocId: null,
  activeDocAnalysis: null,
  backendStatus: "checking",
  
  setPage: (currentPage) => set({ currentPage }),
  toggleTheme: () => set((state) => {
    const nextTheme = state.theme === "light" ? "dark" : "light"
    localStorage.setItem("theme", nextTheme)
    if (nextTheme === "dark") {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
    return { theme: nextTheme }
  }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
  openDocAnalysis: (activeDocId, activeDocAnalysis) => set({ activeDocId, activeDocAnalysis }),
  closeDocAnalysis: () => set({ activeDocId: null, activeDocAnalysis: null }),
  setBackendStatus: (backendStatus) => set({ backendStatus })
}))

