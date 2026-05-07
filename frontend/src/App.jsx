import { useState } from "react";

const API_URL = "http://localhost:8000/predict";

const defaultForm = {
  gestation: "",
  parity: "",
  age: "",
  height: "",
  weight: "",
  smoke: false,
};

const fields = [
  { id: "gestation", label: "Gestation (Weeks)",        placeholder: "Example: 38", min: 20,  max: 45  },
  { id: "parity",    label: "Previous Children (Parity)", placeholder: "Example: 0",  min: 0,   max: 15  },
  { id: "age",       label: "Mother's Age (Years)",          placeholder: "Example: 28", min: 14,  max: 55  },
  { id: "height",    label: "Height (Feet)",            placeholder: "Example: 5.25", min: 4,  max: 8  }, // Height in feet
  { id: "weight",    label: "Weight (Kilograms)",              placeholder: "Example: 60",min: 35,  max: 150 }, // Weight in kg
];

const classConfig = {
  low:    { label: "Low Weight",     range: "< 88 oz (< 2.5 kg)",           bg: "bg-red-50",   border: "border-red-300",   text: "text-red-700",   bar: "bg-red-400"   },
  normal: { label: "Normal",  range: "88–141 oz (2.5–4 kg)",          bg: "bg-green-50", border: "border-green-300", text: "text-green-700", bar: "bg-green-500" },
  high:   { label: "High Weight",   range: "> 141 oz (> 4 kg)",             bg: "bg-blue-50",  border: "border-blue-300",  text: "text-blue-700",  bar: "bg-blue-400"  },
};

export default function App() {
  const [form,    setForm]    = useState(defaultForm);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({ ...prev, [name]: type === "checkbox" ? checked : value }));
  };

  const validate = () =>
    fields.every(({ id }) => form[id] !== "" && !isNaN(parseFloat(form[id])));

  const handleSubmit = async () => {
    setError("");
    if (!validate()) { setError("Please fill in all fields."); return; }

    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(API_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gestation: parseFloat(form.gestation),
          parity:    parseFloat(form.parity),
          age:       parseFloat(form.age),
          height:    parseFloat(form.height) * 12, // Convert feet to inches
          weight:    parseFloat(form.weight) * 2.205, // Convert kg to pounds
          smoke:     form.smoke ? 1 : 0,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError("Server connection failed. Is the Backend running? → " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const probs  = result?.probabilities ?? null;
  const keys   = ["low", "normal", "high"];

  return (
    <div className="min-h-screen bg-gray-50 flex items-start justify-center py-10 px-4">
      <div className="w-full max-w-2xl space-y-5">

        {/* Header */}
        <div>
          <span className="inline-block text-xs px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 font-medium mb-2">
            ML Stacking Ensemble
          </span>
          <h1 className="text-2xl font-semibold text-gray-900">
            Baby Birth Weight Prediction
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            LGBM + XGBoost + Gradient Boosting → Logistic Regression (meta)
          </p>
        </div>

        {/* Input Card */}
        <div className="bg-white rounded-2xl border border-gray-200 p-6 space-y-5">

          {/* Numeric fields */}
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Mother's and Pregnancy Information
            </p>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              {fields.map(({ id, label, placeholder, min, max }) => (
                <div key={id} className="flex flex-col gap-1">
                  <label className="text-xs text-gray-500 font-medium">{label}</label>
                  <input
                    type="number"
                    name={id}
                    value={form[id]}
                    onChange={handleChange}
                    placeholder={placeholder}
                    min={min}
                    max={max}
                    className="border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-900
                               focus:outline-none focus:ring-2 focus:ring-emerald-400 bg-gray-50"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Smoke toggle */}
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Smoking</p>
            <label className="flex items-center gap-3 cursor-pointer w-fit">
              <div className="relative">
                <input
                  type="checkbox"
                  name="smoke"
                  checked={form.smoke}
                  onChange={handleChange}
                  className="sr-only"
                />
                <div className={`w-11 h-6 rounded-full transition-colors ${form.smoke ? "bg-emerald-500" : "bg-gray-200"}`} />
                <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${form.smoke ? "translate-x-5" : ""}`} />
              </div>
              <span className="text-sm text-gray-700">
                {form.smoke ? "Yes (Smokes)" : "No (Does not smoke)"}
              </span>
            </label>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full py-2.5 rounded-xl text-sm font-semibold text-white
                       bg-emerald-600 hover:bg-emerald-700 active:scale-[0.98]
                       disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Analyzing...
              </span>
            ) : "Predict"}
          </button>
        </div>

        {/* Result Card */}
        {result && (
          <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-100">
              <p className="text-sm font-semibold text-gray-800">Prediction Results</p>
              <p className="text-xs text-gray-400 mt-0.5">
                BMI: {result.bmi} kg/m²
              </p>
            </div>

            <div className="px-6 py-5 space-y-5">
              {/* 3 class probability cards */}
              <div className="grid grid-cols-3 gap-3">
                {keys.map((k) => {
                  const cfg      = classConfig[k];
                  const pct      = ((probs[k] ?? 0) * 100).toFixed(1);
                  const isWinner = result.predicted_label === cfg.label;
                  return (
                    <div
                      key={k}
                      className={`rounded-xl border p-3 text-center transition-all
                        ${isWinner ? `${cfg.bg} ${cfg.border} ring-1 ring-offset-1 ${cfg.border.replace("border-","ring-")}` : "border-gray-100 bg-gray-50"}`}
                    >
                      <p className={`text-xs font-medium mb-1 ${isWinner ? cfg.text : "text-gray-500"}`}>
                        {cfg.label}
                      </p>
                      <p className={`text-xl font-semibold ${isWinner ? cfg.text : "text-gray-700"}`}>
                        {pct}%
                      </p>
                    </div>
                  );
                })}
              </div>

              {/* Verdict banner */}
              {(() => {
                const predKey = keys.find(k => classConfig[k].label === result.predicted_label) ?? "normal";
                const cfg = classConfig[predKey];
                return (
                  <div className={`${cfg.bg} ${cfg.border} border rounded-xl px-4 py-3`}>
                    <p className={`text-sm font-semibold ${cfg.text}`}>
                      Prediction: {result.predicted_label}
                    </p>
                    <p className={`text-xs mt-0.5 ${cfg.text} opacity-80`}>
                      {result.predicted_range}
                    </p>
                  </div>
                );
              })()}

              {/* Probability bars */}
              <div className="space-y-2.5">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Probability Distribution
                </p>
                {keys.map((k) => {
                  const cfg = classConfig[k];
                  const pct = ((probs[k] ?? 0) * 100).toFixed(1);
                  return (
                    <div key={k} className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 w-20 shrink-0">{cfg.label}</span>
                      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${cfg.bar} rounded-full transition-all duration-700`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 w-10 text-right">{pct}%</span>
                    </div>
                  );
                })}
              </div>

              <p className="text-xs text-gray-400">
                * This prediction is not a substitute for medical advice.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}