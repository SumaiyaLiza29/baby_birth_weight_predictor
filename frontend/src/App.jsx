import { useState } from "react";

// The relative path ensures Vercel routes the request to the Python backend
const API_URL = "/api/predict";

const defaultForm = {
  gestation: "",
  parity: "",
  age: "",
  height: "",
  weight: "",
  smoke: false,
};

// Input field configurations for the UI
const fields = [
  { id: "gestation", label: "Gestation (Weeks)",         placeholder: "Ex: 38", min: 20,  max: 45  },
  { id: "parity",    label: "Previous Children (Parity)", placeholder: "Ex: 0",  min: 0,   max: 15  },
  { id: "age",       label: "Mother's Age (Years)",          placeholder: "Ex: 28", min: 14,  max: 55  },
  { id: "height",    label: "Height (Feet)",            placeholder: "Ex: 5.25", min: 4,  max: 8   },
  { id: "weight",    label: "Weight (Kilograms)",              placeholder: "Ex: 60", min: 35,  max: 150 },
];

// Styling configuration for prediction result cards
const classConfig = {
  low:    { label: "Low Weight",    range: "< 88 oz (< 2.5 kg)",    bg: "bg-red-50",   border: "border-red-300",   text: "text-red-700",   bar: "bg-red-400"   },
  normal: { label: "Normal",  range: "88–141 oz (2.5–4 kg)",   bg: "bg-green-50", border: "border-green-300", text: "text-green-700", bar: "bg-green-500" },
  high:   { label: "High Weight",   range: "> 141 oz (> 4 kg)",     bg: "bg-blue-50",  border: "border-blue-300",  text: "text-blue-700",  bar: "bg-blue-400"  },
};

export default function App() {
  const [form,    setForm]    = useState(defaultForm);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  // Handle input changes for text, number, and checkboxes
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({ ...prev, [name]: type === "checkbox" ? checked : value }));
  };

  // Basic validation to check if all numeric fields are filled
  const validate = () =>
    fields.every(({ id }) => form[id] !== "" && !isNaN(parseFloat(form[id])));

  // Submit data to the FastAPI backend
  const handleSubmit = async () => {
    setError("");
    if (!validate()) { 
        setError("Please fill in all fields with valid numbers."); 
        return; 
    }

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
          // Converting feet to inches and kg to lbs as per model requirements
          height:    parseFloat(form.height) * 12,    
          weight:    parseFloat(form.weight) * 2.205, 
          smoke:     form.smoke ? 1 : 0,
        }),
      });
      
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError("Server error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const probs  = result?.probabilities ?? null;
  const keys   = ["low", "normal", "high"];

  return (
    <div className="min-h-screen bg-gray-50 flex items-start justify-center py-10 px-4 font-sans">
      <div className="w-full max-w-2xl space-y-6">
        
        {/* Branding Header */}
        <div className="text-center sm:text-left">
          <span className="inline-block text-[10px] uppercase tracking-widest px-3 py-1 rounded-md bg-emerald-100 text-emerald-700 font-bold mb-3">
            AI-Powered Health Analytics
          </span>
          <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">
            Baby Birth Weight Predictor
          </h1>
          <p className="text-gray-500 mt-2 text-sm">
            Input maternal health data to estimate the baby's birth weight category.
          </p>
        </div>

        {/* Main Input Form */}
        <div className="bg-white rounded-3xl shadow-sm border border-gray-100 p-6 sm:p-8 space-y-6">
          <div>
            <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Pregnancy Details</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {fields.map(({ id, label, placeholder }) => (
                <div key={id} className="flex flex-col gap-2">
                  <label className="text-sm text-gray-600 font-semibold">{label}</label>
                  <input
                    type="number"
                    name={id}
                    value={form[id]}
                    onChange={handleChange}
                    placeholder={placeholder}
                    className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:ring-4 focus:ring-emerald-50 transition-all outline-none bg-gray-50/50"
                  />
                </div>
              ))}
              
              {/* Smoking Status */}
              <div className="flex flex-col gap-2">
                <label className="text-sm text-gray-600 font-semibold">Smoking Status</label>
                <div 
                    onClick={() => setForm(f => ({...f, smoke: !f.smoke}))}
                    className={`flex items-center justify-between px-4 py-3 rounded-xl border cursor-pointer transition-all ${form.smoke ? 'bg-emerald-50 border-emerald-200' : 'bg-gray-50 border-gray-200'}`}
                >
                    <span className="text-sm font-medium text-gray-700">{form.smoke ? "Smoker" : "Non-smoker"}</span>
                    <div className={`w-4 h-4 rounded-full border-4 ${form.smoke ? 'bg-emerald-500 border-emerald-200' : 'bg-white border-gray-300'}`} />
                </div>
              </div>
            </div>
          </div>

          {error && (
            <div className="p-4 bg-red-50 text-red-700 text-sm rounded-xl border border-red-100 animate-pulse">
               ⚠️ {error}
            </div>
          )}

          <button 
            onClick={handleSubmit} 
            disabled={loading} 
            className="w-full py-4 bg-gray-900 text-white rounded-2xl font-bold hover:bg-emerald-600 active:scale-95 transition-all shadow-lg shadow-gray-200 disabled:opacity-50"
          >
            {loading ? "Running AI Models..." : "Get Prediction Result"}
          </button>
        </div>

        {/* Prediction Display Section */}
        {result && (
          <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-6 sm:p-8 space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
             <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-gray-800">Analysis Summary</h3>
                <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full">BMI: {result.bmi}</span>
             </div>

             {/* Dynamic Verdict Banner */}
             {(() => {
                const winnerKey = keys.find(k => classConfig[k].label === result.predicted_label);
                const cfg = classConfig[winnerKey];
                return (
                    <div className={`p-5 rounded-2xl border-2 ${cfg.bg} ${cfg.border} text-center`}>
                        <p className={`text-sm font-bold uppercase tracking-widest ${cfg.text} opacity-70`}>Final Verdict</p>
                        <h4 className={`text-3xl font-black ${cfg.text} my-1`}>{result.predicted_label}</h4>
                        <p className={`text-sm font-medium ${cfg.text}`}>{result.predicted_range}</p>
                    </div>
                );
             })()}

             {/* Individual Probability Bars */}
             <div className="space-y-4">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Confidence Scores</p>
                {keys.map(k => {
                   const cfg = classConfig[k];
                   const percentage = (probs[k] * 100).toFixed(1);
                   return (
                     <div key={k} className="space-y-1">
                        <div className="flex justify-between text-xs font-bold text-gray-600">
                            <span>{cfg.label}</span>
                            <span>{percentage}%</span>
                        </div>
                        <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                            <div 
                                className={`h-full ${cfg.bar} rounded-full transition-all duration-1000`} 
                                style={{ width: `${percentage}%` }}
                            />
                        </div>
                     </div>
                   );
                })}
             </div>
             
             <p className="text-[10px] text-gray-400 text-center italic">
                Disclaimer: This AI model provides estimates based on historical data. Consult a doctor for medical diagnosis.
             </p>
          </div>
        )}
      </div>
    </div>
  );
}