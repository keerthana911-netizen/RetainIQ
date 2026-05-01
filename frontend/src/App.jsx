import { useState } from "react";

function App() {
  const [form, setForm] = useState({
    Age: "",
    MonthlyIncome: "",
    OverTime_Yes: 0
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
  };

  const predict = async () => {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        Age: Number(form.Age),
        MonthlyIncome: Number(form.MonthlyIncome),
        OverTime_Yes: Number(form.OverTime_Yes)
      })
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "100vh",
      flexDirection: "column",
      fontFamily: "Arial"
    }}>
      <h1>RetainIQ</h1>

      <input
        name="Age"
        placeholder="Age"
        onChange={handleChange}
        style={{ margin: "5px", padding: "8px" }}
      />

      <input
        name="MonthlyIncome"
        placeholder="Monthly Income"
        onChange={handleChange}
        style={{ margin: "5px", padding: "8px" }}
      />

      <select
        name="OverTime_Yes"
        onChange={handleChange}
        style={{ margin: "5px", padding: "8px" }}
      >
        <option value={0}>No Overtime</option>
        <option value={1}>Overtime</option>
      </select>

      <button onClick={predict} style={{
        marginTop: "10px",
        padding: "10px 20px",
        cursor: "pointer"
      }}>
        Predict
      </button>

      {result && (
        <div style={{ marginTop: "20px", textAlign: "center" }}>
          <h2>Risk Score: {result.risk_score}%</h2>
          <h3 style={{
            color:
              result.risk_level === "High" ? "red" :
              result.risk_level === "Medium" ? "orange" :
              "green"
          }}>
            {result.risk_level}
          </h3>
        </div>
      )}
    </div>
  );
}

export default App;