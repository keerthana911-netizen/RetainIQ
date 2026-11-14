function Field({ f, value, onChange }) {
  if (f.type === 'select' || f.type === 'scale') {
    return (
      <label className="field">
        <span>{f.label}</span>
        <select value={value} onChange={(e) => onChange(f.type === 'scale' ? Number(e.target.value) : e.target.value)}>
          {f.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </label>
    )
  }
  return (
    <label className="field">
      <span>{f.label}<em>{value}</em></span>
      <input type="range" min={f.min} max={f.max} step={f.step} value={value}
             onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

export default function EmployeeForm({ schema, employee, onChange }) {
  const primary = schema.fields.filter((f) => f.primary)
  const advanced = schema.fields.filter((f) => !f.primary)
  const set = (name) => (v) => onChange({ ...employee, [name]: v })
  return (
    <div className="form">
      <div className="grid">
        {primary.map((f) => <Field key={f.name} f={f} value={employee[f.name] ?? f.default} onChange={set(f.name)} />)}
      </div>
      <details>
        <summary>More fields ({advanced.length})</summary>
        <div className="grid">
          {advanced.map((f) => <Field key={f.name} f={f} value={employee[f.name] ?? f.default} onChange={set(f.name)} />)}
        </div>
      </details>
      <p className="muted small">
        Gender, age and marital status are deliberately not model inputs.
      </p>
    </div>
  )
}
