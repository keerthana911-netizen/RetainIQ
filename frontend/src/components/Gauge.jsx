const COLORS = { Low: '#2fbf8f', Medium: '#f5a623', High: '#ff4d5e' }

export default function Gauge({ score, level, size = 220 }) {
  const r = 80, cx = 100, cy = 100
  const pt = (pct) => {
    const a = Math.PI * (1 - pct / 100)
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)]
  }
  const [x1, y1] = pt(0), [x2, y2] = pt(100), [xv, yv] = pt(Math.max(score, 0.5))
  const color = COLORS[level] || '#888'
  return (
    <svg viewBox="0 0 200 125" width={size} role="img" aria-label={`Risk ${score}% (${level})`}>
      <path d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`} className="gauge-track" />
      <path d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${xv} ${yv}`} stroke={color} className="gauge-fill" />
      <text x="100" y="92" textAnchor="middle" className="gauge-num">{Math.round(score)}%</text>
      <text x="100" y="114" textAnchor="middle" className="gauge-lvl" fill={color}>{level} risk</text>
    </svg>
  )
}
export { COLORS }
