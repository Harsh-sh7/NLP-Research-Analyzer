import { create } from "zustand"

interface User {
  id: string
  email: string
  username?: string | null
  created_at: string
}

interface AuthState {
  token: string | null
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  setToken: (token: string | null) => void
  setUser: (user: User | null) => void
  login: (token: string, user: User) => void
  logout: () => void
  setError: (error: string | null) => void
  setLoading: (loading: boolean) => void
}

export const useAuthStore = create<AuthState>((set) => ({
  token: localStorage.getItem("token"),
  user: null,
  isAuthenticated: !!localStorage.getItem("token"),
  isLoading: false,
  error: null,
  setToken: (token) => {
    if (token) {
      localStorage.setItem("token", token)
    } else {
      localStorage.removeItem("token")
    }
    set({ token, isAuthenticated: !!token })
  },
  setUser: (user) => set({ user }),
  login: (token, user) => {
    localStorage.setItem("token", token)
    set({ token, user, isAuthenticated: true, error: null })
  },
  logout: () => {
    localStorage.removeItem("token")
    set({ token: null, user: null, isAuthenticated: false, error: null })
  },
  setError: (error) => set({ error }),
  setLoading: (isLoading) => set({ isLoading }),
}))
