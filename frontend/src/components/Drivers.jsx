import { BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const UP = '#ff4d5e', DOWN = '#2fbf8f'

function Tip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="tip">
      <b>{d.label}: {d.value}</b>
      <div>{d.odds_multiplier >= 1
        ? `Multiplies the odds of leaving by ×${d.odds_multiplier.toFixed(2)}`
        : `Cuts the odds of leaving to ×${d.odds_multiplier.toFixed(2)}`}</div>
    </div>
  )
}

export default function Drivers({ explanation }) {
  const rows = explanation.drivers.map((d) => ({ ...d, name: `${d.label}: ${d.value}` }))
  const ups = explanation.drivers.filter((d) => d.contribution > 0).slice(0, 3)
  const downs = explanation.drivers.filter((d) => d.contribution < 0).slice(0, 2)
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 0.5)
  return (
    <div>
      <p className="lead">
        {ups.length > 0 && <>Pushing risk <b className="up">up</b>: {ups.map((d, i) => (
          <span key={d.feature}>{i ? ', ' : ''}{d.label.toLowerCase()} ({d.value}, ×{d.odds_multiplier.toFixed(1)})</span>))}. </>}
        {downs.length > 0 && <>Holding it <b className="down">down</b>: {downs.map((d, i) => (
          <span key={d.feature}>{i ? ', ' : ''}{d.label.toLowerCase()} ({d.value})</span>))}.</>}
      </p>
      <ResponsiveContainer width="100%" height={rows.length * 34 + 30}>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
          <XAxis type="number" domain={[-max * 1.1, max * 1.1]} hide />
          <YAxis type="category" dataKey="name" width={215} tick={{ fill: '#c9d1dc', fontSize: 12 }} axisLine={false} tickLine={false} />
          <ReferenceLine x={0} stroke="#46505e" />
          <Tooltip content={<Tip />} cursor={{ fill: 'rgba(255,255,255,.04)' }} />
          <Bar dataKey="contribution" radius={3} barSize={16}>
            {rows.map((r) => <Cell key={r.feature} fill={r.contribution > 0 ? UP : DOWN} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="muted small">
        An employee with a typical profile starts at {explanation.base_risk.toFixed(1)}%; the factors above move this
        employee to {explanation.final_risk.toFixed(1)}%. Bars are SHAP contributions (log-odds); hover for the odds multiplier.
        {explanation.other_features.count > 0 && ` ${explanation.other_features.count} smaller factors are not shown.`}
      </p>
    </div>
  )
}
