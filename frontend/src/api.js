import { useEffect, useState } from 'react'

// In production behind nginx (docker-compose) the API lives at /api.
// On Vercel, set VITE_API_URL to the backend's public URL.
const BASE = (import.meta.env.VITE_API_URL ?? '/api').replace(/\/$/, '')

async function request(path, body) {
  const res = await fetch(BASE + path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`)
  return data
}

export const api = {
  schema: () => request('/schema'),
  personas: () => request('/personas'),
  modelCard: () => request('/model-card'),
  predict: (employee) => request('/predict', employee),
  whatif: (employee, changes) => request('/whatif', { employee, changes }),
  sweep: (employee, feature) => request('/sweep', { employee, feature }),
}

export function useDebounced(value, ms = 250) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}
