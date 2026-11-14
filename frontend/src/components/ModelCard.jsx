import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const f = (x, d = 3) => (x === undefined || x === null ? '–' : Number(x).toFixed(d))
const ci = (a) => (a ? `${f(a[0], 2)}–${f(a[1], 2)}` : '')

export default function ModelCard({ card }) {
  const m = card.metrics
  const order = ['Logistic Regression', 'Random Forest', 'XGBoost']
  const champ = m.champion
  const cm = m.models[champ]
  const ab = cm.imbalance_ablation
  const bands = m.risk_bands.test
  return (
    <div className="stack">
      <section className="card">
        <h2>Why not just accuracy?</h2>
        <p>
          Only {(m.dataset.attrition_rate * 100).toFixed(0)}% of employees leave, so a model that predicts
          “stays” for everyone is already <b>{(m.majority_baseline_accuracy * 100).toFixed(0)}% accurate</b> while
          catching zero leavers. RetainIQ is judged on ROC-AUC, PR-AUC, precision and recall instead.
        </p>
      </section>

      <section className="card">
        <h2>Model comparison <span className="muted small">held-out test set · {m.dataset.test_rows} employees, {m.dataset.test_positives} leavers</span></h2>
        <div className="scroll">
          <table>
            <thead>
              <tr><th>Model</th><th>CV ROC-AUC</th><th>Test ROC-AUC <small>(95% CI)</small></th><th>Test PR-AUC</th>
                <th>Precision</th><th>Recall</th><th>F1</th></tr>
            </thead>
            <tbody>
              {order.map((n) => {
                const r = m.models[n], t = r.test, b = t.at_balanced_threshold
                return (
                  <tr key={n} className={n === champ ? 'champ' : ''}>
                    <td>{n}{n === champ && <span className="badge">deployed</span>}</td>
                    <td>{f(r.cv.roc_auc[0])} <small>±{f(r.cv.roc_auc[1], 2)}</small></td>
                    <td>{f(t.roc_auc)} <small>({ci(t.roc_auc_ci95)})</small></td>
                    <td>{f(t.pr_auc)}</td><td>{f(b.precision, 2)}</td><td>{f(b.recall, 2)}</td><td>{f(b.f1, 2)}</td>
                  </tr>
                )
              })}
              <tr className="base"><td>Baseline (“always stays”)</td><td>0.500</td><td>0.500</td><td>{f(m.models["Baseline (always 'stays')"].test.pr_auc)}</td><td>0</td><td>0</td><td>0</td></tr>
            </tbody>
          </table>
        </div>
        <p className="muted small">
          Precision/recall/F1 are at each model's own F1-optimal threshold chosen on out-of-fold training data (never the test set).
          The differences between Logistic Regression and XGBoost are well inside the confidence intervals: with 47 test leavers, treat them as tied.
          {' '}{m.champion_rule}
        </p>
      </section>

      <div className="split">
        <section className="card">
          <h2>Handling imbalance</h2>
          <p className="small">Same tuned {champ}, with and without class weighting (CV, threshold 0.5):</p>
          <table className="compact">
            <thead><tr><th></th><th>ROC-AUC</th><th>Recall</th><th>Precision</th></tr></thead>
            <tbody>
              <tr><td>No weighting</td><td>{f(ab.without_class_weight.cv_roc_auc[0], 2)}</td><td>{f(ab.without_class_weight['cv_recall@0.5'][0], 2)}</td><td>{f(ab.without_class_weight['cv_precision@0.5'][0], 2)}</td></tr>
              <tr><td>Class-weighted</td><td>{f(ab.with_class_weight.cv_roc_auc[0], 2)}</td><td>{f(ab.with_class_weight['cv_recall@0.5'][0], 2)}</td><td>{f(ab.with_class_weight['cv_precision@0.5'][0], 2)}</td></tr>
            </tbody>
          </table>
          <p className="muted small">Weighting doesn't improve ranking (AUC); it moves the operating point to catch many more leavers. Scores are then calibrated so “40%” means ≈40%.</p>
        </section>
        <section className="card">
          <h2>Do the risk bands mean anything?</h2>
          <table className="compact">
            <thead><tr><th>Band</th><th>% of staff</th><th>Actually left</th></tr></thead>
            <tbody>
              {['Low', 'Medium', 'High'].map((b) => (
                <tr key={b}><td>{b}</td><td>{bands[b].share_of_employees_pct}%</td><td><b>{bands[b].observed_attrition_pct}%</b></td></tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">Held-out test set. Overall attrition: {(m.dataset.attrition_rate * 100).toFixed(1)}%.</p>
        </section>
      </div>

      <section className="card">
        <h2>What drives risk overall <span className="muted small">mean |SHAP|, all employees</span></h2>
        <ResponsiveContainer width="100%" height={card.global_importance.length * 28 + 20}>
          <BarChart data={card.global_importance} layout="vertical" margin={{ left: 10, right: 20 }}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="label" width={190} tick={{ fill: '#c9d1dc', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} formatter={(v) => [v, 'mean |SHAP|']} />
            <Bar dataKey="mean_abs_shap" fill="#ff4d5e" radius={3} barSize={14} />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section className="card">
        <h2>Limits you should know about</h2>
        <ul className="small">
          <li>Trained on the IBM HR Analytics dataset, which is <b>fictional and synthetic</b>. This is a demonstration, not a validated HR tool.</li>
          <li>The model finds correlations, not causes. What-if results are model sensitivities, not promises.</li>
          <li>Gender, age and marital status are excluded. Excluding them cost {f(m.sensitive_feature_ablation.cv_roc_auc_with[0] - m.sensitive_feature_ablation.cv_roc_auc_without[0], 3)} AUC.</li>
          <li>Scores should start conversations with employees, never decide pay, promotion or dismissal.</li>
        </ul>
      </section>
    </div>
  )
}
