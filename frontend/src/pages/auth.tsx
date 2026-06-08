import React, { useState } from "react"
import { useAuthStore } from "@/store/authStore"
import { useUIStore } from "@/store/uiStore"
import { api } from "@/lib/api"
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Brain, AlertCircle, ArrowLeft } from "lucide-react"

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState("")
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  
  const loginStore = useAuthStore((state) => state.login)
  const setPage = useUIStore((state) => state.setPage)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg(null)
    setIsLoading(true)

    try {
      if (isLogin) {
        // Log in
        const res = await api.auth.login(email, password)
        localStorage.setItem("token", res.access_token)
        const userDetails = await api.auth.getMe()
        loginStore(res.access_token, userDetails)
        setPage("research") // Go directly to agent workspace after login
      } else {
        // Sign up
        await api.auth.signup(email, password, username)
        // Auto log in after sign up
        const res = await api.auth.login(email, password)
        localStorage.setItem("token", res.access_token)
        const userDetails = await api.auth.getMe()
        loginStore(res.access_token, userDetails)
        setPage("research") // Go directly to agent workspace after signup
      }
    } catch (err: any) {
      setErrorMsg(err.message || "An authentication error occurred. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex-1 min-h-[calc(100vh-80px)] flex flex-col justify-center items-center px-6 relative py-12">
      {/* Background decoration */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-blue-500/10 dark:bg-blue-600/10 rounded-full blur-[100px] pointer-events-none -z-10" />

      {/* Header Back CTA */}
      <button 
        className="absolute top-0 left-6 flex items-center gap-2 cursor-pointer border-0 bg-transparent text-zinc-500 hover:text-zinc-900 dark:hover:text-white transition-colors py-2 px-1 text-xs font-semibold" 
        onClick={() => setPage("landing")}
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Back to Home</span>
      </button>

      <Card className="w-full max-w-md border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-4 shadow-xl">
        <CardHeader className="space-y-1 text-center">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center mx-auto mb-4 shadow-md shadow-blue-500/20">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <CardTitle className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-white">
            {isLogin ? "Welcome Back" : "Create Account"}
          </CardTitle>
          <CardDescription className="text-zinc-500 dark:text-zinc-400 text-xs">
            {isLogin
              ? "Enter your credentials to access the workspace"
              : "Sign up to start deploying research agents"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {errorMsg && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div className="space-y-1 animate-in fade-in-50 duration-200">
                <label className="text-xs font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">Username</label>
                <Input
                  type="text"
                  placeholder="johndoe"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  className="border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 focus:border-blue-500/50"
                />
              </div>
            )}

            <div className="space-y-1">
              <label className="text-xs font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">Email Address</label>
              <Input
                type="email"
                placeholder="name@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 focus:border-blue-500/50"
              />
            </div>
            
            <div className="space-y-1">
              <label className="text-xs font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">Password</label>
              <Input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 focus:border-blue-500/50"
              />
            </div>

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full mt-2 font-semibold bg-gradient-to-r from-blue-600 to-indigo-500 hover:from-blue-500 hover:to-indigo-400 text-white shadow-md shadow-blue-500/10 cursor-pointer"
            >
              {isLoading ? "Please wait..." : isLogin ? "Sign In" : "Sign Up"}
            </Button>
          </form>

          <div className="text-center text-xs text-zinc-500 mt-4">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button
              onClick={() => {
                setIsLogin(!isLogin)
                setErrorMsg(null)
              }}
              className="text-blue-600 dark:text-blue-400 hover:underline font-bold bg-transparent border-0 cursor-pointer p-0"
            >
              {isLogin ? "Sign Up" : "Sign In"}
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
