import { useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend
} from "recharts";

import "./App.css";

function App() {
  const [result, setResult] = useState(null);

  const predictRisk = async () => {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        Age: 18,
        MonthlyIncome: 90000,
        OverTime_Yes: 1,
      }),
    });

    const data = await response.json();
    setResult(data);
  };

  const chartData = result
    ? [
        { name: "Risk", value: result.risk_score },
        { name: "Safe", value: 100 - result.risk_score },
      ]
    : [];

  const COLORS = ["#ff4d4f", "#00C49F"];

  return (
    <div className="app">
      <h1>RetainIQ Dashboard</h1>

      <button onClick={predictRisk}>
        Predict Attrition Risk
      </button>

      {result && (
        <>
          <h2>Risk Score: {result.risk_score}%</h2>
          <h3>{result.risk_level}</h3>

          <PieChart width={400} height={300}>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              outerRadius={100}
              dataKey="value"
              label
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Pie>

            <Tooltip />
            <Legend />
          </PieChart>
        </>
      )}
    </div>
  );
}

export default App;