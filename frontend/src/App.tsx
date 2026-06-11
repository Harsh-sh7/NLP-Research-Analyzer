import React, { useEffect, useState } from "react"
import { useUIStore } from "@/store/uiStore"
import { useAuthStore } from "@/store/authStore"
import { api } from "@/lib/api"
import LandingPage from "@/pages/landing"
import AuthPage from "@/pages/auth"
import NLPWorkspace from "@/pages/nlp"
import ResearchWorkspace from "@/pages/research"
import HistoryTimeline from "@/pages/history"
import { 
  Brain, 
  Sparkles, 
  History, 
  Sun, 
  Moon, 
  LogOut,
  LogIn,
  Home,
  Loader2
} from "lucide-react"

export default function App() {
  const { currentPage, setPage, theme, toggleTheme, backendStatus, setBackendStatus } = useUIStore()
  const { user, setUser, token, isAuthenticated, logout } = useAuthStore()

  // Dynamic Cursor Glow position tracking
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 })
  const [isVisible, setIsVisible] = useState(false)


  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY })
      setIsVisible(true)
    }

    const handleMouseLeave = () => {
      setIsVisible(false)
    }

    const handleMouseEnter = () => {
      setIsVisible(true)
    }

    window.addEventListener("mousemove", handleMouseMove)
    document.addEventListener("mouseleave", handleMouseLeave)
    document.addEventListener("mouseenter", handleMouseEnter)

    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseleave", handleMouseLeave)
      document.removeEventListener("mouseenter", handleMouseEnter)
    }
  }, [])

  // Ping backend to wake it up and keep it warm
  useEffect(() => {
    const backendRoot = (import.meta.env.VITE_API_URL || "http://localhost:8000/api").replace(/\/api$/, "")
    let isMounted = true
    let sleepingTimeout: any

    const pingBackend = async () => {
      const currentStatus = useUIStore.getState().backendStatus
      // Show "waking up" warning if backend doesn't respond in 1.5 seconds
      if (currentStatus !== "online") {
        sleepingTimeout = setTimeout(() => {
          if (isMounted) setBackendStatus("sleeping")
        }, 1500)
      }

      try {
        const res = await fetch(`${backendRoot}/`, { mode: "cors" })
        if (res.ok) {
          if (isMounted) {
            setBackendStatus("online")
            clearTimeout(sleepingTimeout)
          }
        } else {
          throw new Error("Backend offline")
        }
      } catch (err) {
        console.warn("Backend connection check failed:", err)
        if (isMounted) {
          const latestStatus = useUIStore.getState().backendStatus
          // If we were already in "sleeping" state, keep it. Otherwise, flag as connection error.
          setBackendStatus(latestStatus === "sleeping" ? "sleeping" : "error")
          clearTimeout(sleepingTimeout)
        }
      }
    }

    pingBackend()

    // Keep-alive ping every 3 minutes to prevent Render spin-down
    const interval = setInterval(() => {
      pingBackend()
    }, 180000)

    return () => {
      isMounted = false
      clearTimeout(sleepingTimeout)
      clearInterval(interval)
    }
  }, [])

  // Auto-validate current session token on boot
  useEffect(() => {
    const validateToken = async () => {
      if (token && !user) {
        try {
          const userDetails = await api.auth.getMe()
          setUser(userDetails)
        } catch (err) {
          console.error("Token expired or invalid", err)
          logout()
        }
      }
    }
    validateToken()
  }, [token, user, setUser, logout])

  // Initialize and update HTML class on theme change
  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
  }, [theme])

  const handleLogout = () => {
    logout()
    setPage("landing")
  }

  const renderActivePage = () => {
    switch (currentPage) {
      case "landing": return <LandingPage />
      case "auth": return <AuthPage />
      case "nlp": return <NLPWorkspace />
      case "research": return <ResearchWorkspace />
      case "history": return <HistoryTimeline />
      default: return <LandingPage />
    }
  }

  // Hide global header on landing/auth if they want standalone landing look,
  // but to keep it unified, let's render the floating navbar on nlp, research, history.
  const showNavbar = currentPage !== "auth";

  return (
    <div className="min-h-screen flex flex-col text-foreground bg-background transition-colors duration-300">
      {showNavbar && (
        <div className="w-full px-4 pt-4 sticky top-0 z-50">
          <header className="max-w-5xl mx-auto px-6 py-3 bg-white/70 dark:bg-zinc-950/70 border border-zinc-200/50 dark:border-zinc-800/50 backdrop-blur-md rounded-full flex items-center justify-between shadow-sm transition-all">
            {/* Logo */}
            <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => setPage("landing")}>
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-md shadow-blue-500/20">
                <Brain className="w-4.5 h-4.5 text-white" />
              </div>
              <span className="font-bold text-sm tracking-tight bg-gradient-to-r from-zinc-900 to-zinc-600 dark:from-white dark:to-zinc-300 bg-clip-text text-transparent">
                NLP Swarm
              </span>
            </div>

            {/* Navigation Navigation Pills */}
            <nav className="flex items-center gap-1">
              <button
                onClick={() => setPage("landing")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all cursor-pointer ${
                  currentPage === "landing"
                    ? "bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 shadow-sm"
                    : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
                }`}
              >
                <Home className="w-3.5 h-3.5" />
                <span>Home</span>
              </button>
              
              <button
                onClick={() => setPage(isAuthenticated ? "nlp" : "auth")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all cursor-pointer ${
                  currentPage === "nlp"
                    ? "bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 shadow-sm"
                    : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
                }`}
              >
                <Brain className="w-3.5 h-3.5" />
                <span>Classical NLP</span>
              </button>

              <button
                onClick={() => setPage(isAuthenticated ? "research" : "auth")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all cursor-pointer ${
                  currentPage === "research"
                    ? "bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 shadow-sm"
                    : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Agentic Swarm</span>
              </button>

              <button
                onClick={() => setPage(isAuthenticated ? "history" : "auth")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all cursor-pointer ${
                  currentPage === "history"
                    ? "bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 shadow-sm"
                    : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
                }`}
              >
                <History className="w-3.5 h-3.5" />
                <span>History</span>
              </button>
            </nav>

            {/* Right CTAs */}
            <div className="flex items-center gap-3">
              <button
                onClick={toggleTheme}
                className="text-zinc-500 hover:text-zinc-900 dark:hover:text-white p-1.5 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors cursor-pointer"
                title="Toggle Theme"
              >
                {theme === "light" ? <Moon className="w-4.5 h-4.5" /> : <Sun className="w-4.5 h-4.5" />}
              </button>

              {isAuthenticated ? (
                <div className="flex items-center gap-3">
                  <span className="hidden sm:inline text-xs text-zinc-500 font-medium truncate max-w-[120px]" title={user?.username || user?.email}>
                    {user?.username || user?.email}
                  </span>
                  <button
                    onClick={handleLogout}
                    className="text-zinc-500 hover:text-destructive p-1.5 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors cursor-pointer"
                    title="Log Out"
                  >
                    <LogOut className="w-4.5 h-4.5" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setPage("auth")}
                  className="px-4 py-1.5 rounded-full text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-colors flex items-center gap-1 cursor-pointer"
                >
                  <LogIn className="w-3.5 h-3.5" />
                  <span>Launch App</span>
                </button>
              )}
            </div>
          </header>
        </div>
      )}

      {/* Dynamic Cursor Glow Effect */}
      <div 
        className="pointer-events-none fixed top-0 left-0 w-[450px] h-[450px] rounded-full z-[9999] transition-opacity duration-300"
        style={{
          opacity: isVisible ? 1 : 0,
          transform: `translate3d(calc(${mousePos.x}px - 50%), calc(${mousePos.y}px - 50%), 0)`,
          transition: "transform 0.08s cubic-bezier(0.1, 0.8, 0.25, 1), opacity 0.3s ease",
          background: theme === "dark" 
            ? "radial-gradient(circle, rgba(59, 130, 246, 0.12) 0%, rgba(99, 102, 241, 0.04) 45%, rgba(0, 0, 0, 0) 70%)"
            : "radial-gradient(circle, rgba(59, 130, 246, 0.07) 0%, rgba(99, 102, 241, 0.02) 45%, rgba(0, 0, 0, 0) 70%)",
        }}
      />

      {/* Backend Spin-up Notifier */}
      {backendStatus === "sleeping" && (
        <div className="fixed bottom-6 right-6 z-[99999] max-w-sm p-4 rounded-2xl border border-zinc-200/50 dark:border-zinc-800/50 bg-white/85 dark:bg-zinc-950/85 backdrop-blur-xl shadow-xl shadow-blue-500/5 flex gap-3.5 items-start transition-all duration-300">
          <div className="w-9 h-9 rounded-xl bg-blue-500/10 flex items-center justify-center shrink-0 border border-blue-500/20 text-blue-600 dark:text-blue-400">
            <Loader2 className="w-4.5 h-4.5 animate-spin" />
          </div>
          <div className="flex-1">
            <h4 className="text-xs font-semibold text-zinc-900 dark:text-white">Connecting to Service...</h4>
            <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-1 leading-relaxed">
              This app is waking up from a deep sleep on Render's free tier. This typically takes 30-50 seconds.
            </p>
            <div className="w-full bg-zinc-100 dark:bg-zinc-900 h-1 rounded-full mt-2.5 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-500 to-indigo-500 h-full rounded-full animate-[loading_45s_ease-out_infinite]" />
            </div>
          </div>
        </div>
      )}

      {backendStatus === "error" && (
        <div className="fixed bottom-6 right-6 z-[99999] max-w-sm p-4 rounded-2xl border border-red-200/50 dark:border-red-900/50 bg-red-50/90 dark:bg-red-950/40 backdrop-blur-xl shadow-xl flex gap-3.5 items-start transition-all duration-300 animate-in fade-in slide-in-from-bottom-5">
          <div className="w-9 h-9 rounded-xl bg-red-500/10 flex items-center justify-center shrink-0 border border-red-500/20 text-red-600 dark:text-red-400">
            <span className="text-sm">⚠️</span>
          </div>
          <div className="flex-1">
            <h4 className="text-xs font-semibold text-red-900 dark:text-red-200">Connection Error</h4>
            <p className="text-[11px] text-red-700/80 dark:text-red-400/80 mt-1 leading-relaxed">
              Could not reach the backend. If you recently deployed, check your Render CORS `FRONTEND_URL` variable, or wait a minute for startup.
            </p>
          </div>
        </div>
      )}

      {/* Main Content Viewport */}
      <main className="flex-1 flex flex-col relative">
        {renderActivePage()}
      </main>
    </div>
  )
}
