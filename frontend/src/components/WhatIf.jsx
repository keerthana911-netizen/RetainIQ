import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceDot, CartesianGrid } from 'recharts'
import { api, useDebounced } from '../api'
import Gauge from './Gauge'

// Levers HR can actually pull (or at least discuss).
const LEVERS = ['OverTime', 'MonthlyIncome', 'StockOptionLevel', 'BusinessTravel', 'WorkLifeBalance',
  'JobSatisfaction', 'YearsSinceLastPromotion', 'TrainingTimesLastYear']
const CURVES = ['MonthlyIncome', 'YearsSinceLastPromotion', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany']

const label = (f, v) => f.options?.find((o) => String(o.value) === String(v))?.label ?? v

function Lever({ f, current, value, onChange }) {
  const changed = value !== undefined && value !== current
  const v = value ?? current
  return (
    <label className={`field lever ${changed ? 'changed' : ''}`}>
      <span>{f.label}<em>{changed ? `${label(f, current)} → ${label(f, v)}` : `${label(f, current)}`}</em></span>
      {f.type === 'number' ? (
        <input type="range" min={f.min} max={f.max} step={f.step} value={v} onChange={(e) => onChange(Number(e.target.value))} />
      ) : (
        <select value={v} onChange={(e) => onChange(f.type === 'scale' ? Number(e.target.value) : e.target.value)}>
          {f.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      )}
    </label>
  )
}

export default function WhatIf({ schema, employee, current }) {
  const fields = Object.fromEntries(schema.fields.map((f) => [f.name, f]))
  const [changes, setChanges] = useState({})
  const [result, setResult] = useState(null)
  const [curveFeature, setCurveFeature] = useState('MonthlyIncome')
  const [curve, setCurve] = useState(null)
  const [error, setError] = useState('')

  const actual = (name) => current.inputs_used[name]
  const active = Object.fromEntries(Object.entries(changes).filter(([k, v]) => v !== actual(k)))
  const dChanges = useDebounced(active, 250)
  const key = JSON.stringify([employee, dChanges])

  useEffect(() => {
    if (!Object.keys(dChanges).length) return
    let live = true
    api.whatif(employee, dChanges).then((r) => live && (setResult(r), setError(''))).catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [key]) // eslint-disable-line

  const dEmp = useDebounced(employee, 300)
  useEffect(() => {
    let live = true
    api.sweep(dEmp, curveFeature).then((r) => live && setCurve(r)).catch(() => {})
    return () => { live = false }
  }, [dEmp, curveFeature])

  const hasChanges = Object.keys(active).length > 0
  const shown = hasChanges ? result : null   // derived, so no setState-in-effect
  const scen = shown?.scenario
  const scenarioValueOfCurve = changes[curveFeature] ?? undefined

  return (
    <div className="split">
      <section className="card">
        <h2>Change something</h2>
        <p className="muted small">Adjust the levers below; risk updates live. Compared against the employee on the first tab.</p>
        <div className="grid one">
          {LEVERS.filter((n) => fields[n]).map((n) => (
            <Lever key={n} f={fields[n]} current={actual(n)} value={changes[n]}
                   onChange={(v) => setChanges((c) => ({ ...c, [n]: v }))} />
          ))}
        </div>
        <button className="ghost" onClick={() => setChanges({})} disabled={!Object.keys(changes).length}>Reset scenario</button>
      </section>

      <section className="card">
        <h2>Before vs after</h2>
        {error && <p className="error">{error}</p>}
        <div className="ba">
          <div><p className="muted small center">Current</p><Gauge score={current.risk_score} level={current.risk_level} size={190} /></div>
          <div className="arrow">→</div>
          <div>
            <p className="muted small center">Scenario</p>
            {scen ? <Gauge score={scen.risk_score} level={scen.risk_level} size={190} />
                  : <div className="placeholder">Change a lever</div>}
          </div>
        </div>
        {shown && (
          <>
            <p className={`delta ${shown.delta_pp < 0 ? 'down' : shown.delta_pp > 0 ? 'up' : ''}`}>
              {shown.delta_pp > 0 ? '+' : ''}{shown.delta_pp} percentage points
            </p>
            {shown.driver_shifts.length > 0 && (
              <p className="muted small center">Biggest model shifts: {shown.driver_shifts.slice(0, 3).map((s) => s.label).join(', ')}</p>
            )}
            {shown.clipped_inputs.length > 0 && <p className="muted small center">Values outside the training range were clipped.</p>}
          </>
        )}
        <h3>Sensitivity curve</h3>
        <select value={curveFeature} onChange={(e) => setCurveFeature(e.target.value)}>
          {CURVES.filter((n) => fields[n]).map((n) => <option key={n} value={n}>{fields[n].label}</option>)}
        </select>
        {curve && (
          <ResponsiveContainer width="100%" height={210}>
            <LineChart data={curve.points} margin={{ top: 14, right: 16, bottom: 6, left: -8 }}>
              <CartesianGrid stroke="#2a313c" vertical={false} />
              <XAxis dataKey="value" type="number" domain={['dataMin', 'dataMax']} tick={{ fill: '#9aa5b4', fontSize: 11 }} />
              <YAxis unit="%" domain={[0, 100]} tick={{ fill: '#9aa5b4', fontSize: 11 }} />
              <Tooltip formatter={(v) => [`${v}%`, 'Risk']} labelFormatter={(l) => `${curve.label}: ${l}`}
                       contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Line type="monotone" dataKey="risk_score" stroke="#ff4d5e" strokeWidth={2} dot={false} />
              <ReferenceDot x={curve.current} y={current.risk_score} r={6} fill="#fff" stroke="#ff4d5e" />
              {scenarioValueOfCurve !== undefined && scen && (
                <ReferenceDot x={scenarioValueOfCurve} y={scen.risk_score} r={6} fill="#2fbf8f" stroke="#fff" />)}
            </LineChart>
          </ResponsiveContainer>
        )}
        <p className="muted small">All other inputs held fixed. {shown?.caveat ?? 'The model learned correlations in historical data; a change here shows what the model associates with different inputs, not a guaranteed causal effect.'}</p>
      </section>
    </div>
  )
}
