import { useEffect, useState } from 'react'
import { api, useDebounced } from './api'
import EmployeeForm from './components/EmployeeForm'
import Gauge from './components/Gauge'
import Drivers from './components/Drivers'
import WhatIf from './components/WhatIf'
import ModelCard from './components/ModelCard'
import './App.css'

const TABS = [['risk', 'Risk & why'], ['whatif', 'What-if'], ['model', 'Model']]

export default function App() {
  const [tab, setTab] = useState('risk')
  const [schema, setSchema] = useState(null)
  const [personas, setPersonas] = useState({})
  const [card, setCard] = useState(null)
  const [employee, setEmployee] = useState({})
  const [persona, setPersona] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [booting, setBooting] = useState(true)

  // Load schema/personas/model card. Free hosts sleep, so retry while the API wakes up.
  useEffect(() => {
    let stop = false
    const boot = async (attempt = 0) => {
      try {
        const [s, p, c] = await Promise.all([api.schema(), api.personas(), api.modelCard()])
        if (stop) return
        setSchema(s); setPersonas(p); setCard(c)
        const first = Object.keys(p)[0]
        setPersona(first); setEmployee(p[first]); setBooting(false)
      } catch {
        if (stop) return
        if (attempt < 12) setTimeout(() => boot(attempt + 1), 4000)
        else { setError('Could not reach the RetainIQ API.'); setBooting(false) }
      }
    }
    boot()
    return () => { stop = true }
  }, [])

  const dEmployee = useDebounced(employee, 250)
  useEffect(() => {
    if (!schema) return
    let live = true
    api.predict(dEmployee).then((r) => { if (live) { setResult(r); setError('') } })
      .catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [dEmployee, schema])

  const pick = (name) => { setPersona(name); setEmployee(personas[name]) }

  return (
    <div className="app">
      <header>
        <div className="brand"><span className="dot" /> RetainIQ</div>
        <p className="tag">Explainable attrition risk: see <i>who</i> might leave and <i>why</i>.</p>
        <nav>
          {TABS.map(([k, label]) => (
            <button key={k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{label}</button>
          ))}
        </nav>
      </header>

      {booting && <div className="card center">Waking up the API… <span className="muted small">(free hosting sleeps when idle; this can take ~30s)</span></div>}
      {error && <div className="card error">{error}</div>}

      {schema && tab === 'risk' && (
        <div className="split">
          <section className="card">
            <h2>Employee</h2>
            <div className="chips">
              {Object.keys(personas).map((n) => (
                <button key={n} className={`chip ${persona === n ? 'on' : ''}`} onClick={() => pick(n)}>{n}</button>
              ))}
            </div>
            <EmployeeForm schema={schema} employee={employee} onChange={(e) => { setEmployee(e); setPersona('') }} />
          </section>
          <section className="card result-first">
            <h2>Attrition risk</h2>
            {result ? (
              <>
                <div className="center"><Gauge score={result.risk_score} level={result.risk_level} /></div>
                <p className="center muted">
                  Riskier than <b>{result.riskier_than_pct}%</b> of employees · company baseline {result.model.base_rate_pct}%
                </p>
                <h3>Why this score</h3>
                <Drivers explanation={result.explanation} />
                {result.assumed_defaults.length > 0 && (
                  <p className="muted small">{result.assumed_defaults.length} fields not provided were set to typical values.</p>
                )}
              </>
            ) : <p className="muted">Scoring…</p>}
          </section>
        </div>
      )}

      {schema && tab === 'whatif' && result && <WhatIf schema={schema} employee={employee} current={result} />}
      {tab === 'model' && card && <ModelCard card={card} />}

      <footer className="muted small">
        Demo built on the fictional IBM HR dataset. Use scores to start supportive conversations, not to make decisions about people.
      </footer>
    </div>
  )
}
