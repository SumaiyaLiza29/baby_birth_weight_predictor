# শিশুর জন্মকালীন ওজন পূর্বাভাস — সম্পূর্ণ প্রজেক্ট

## ফোল্ডার কাঠামো

```
baby_bwt_predictor/
├── data/
│   └── final_continuous_babies_data.csv   ← আপনার ডেটা এখানে রাখুন
├── models/                                ← train করলে এখানে .pkl সেভ হবে
├── backend/
│   ├── main.py                            ← FastAPI সার্ভার
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                        ← React UI
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── train.py                               ← মডেল ট্রেনিং স্ক্রিপ্ট
└── README.md
```

---

## ধাপ ১ — মডেল ট্রেন করুন

```bash
# প্রজেক্ট ফোল্ডারে যান
cd baby_bwt_predictor

# Python প্যাকেজ ইনস্টল (একবার)
pip install -r backend/requirements.txt

# ডেটা রাখুন: data/final_continuous_babies_data.csv
# তারপর ট্রেন করুন
python train.py
```

এটি চালালে `models/` ফোল্ডারে তৈরি হবে:
- `lgbm.pkl`
- `xgb.pkl`
- `gb.pkl`
- `meta_model.pkl`
- `scaler.pkl`

---

## ধাপ ২ — Backend চালু করুন

```bash
cd baby_bwt_predictor
uvicorn backend.main:app --reload --port 8000
```

চেক করুন: http://localhost:8000/health
API ডক: http://localhost:8000/docs

---

## ধাপ ৩ — Frontend চালু করুন

নতুন টার্মিনালে:

```bash
cd baby_bwt_predictor/frontend
npm install
npm run dev
```

Browser-এ যান: http://localhost:3000

---

## API ব্যবহার (সরাসরি)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gestation": 38,
    "parity": 0,
    "age": 28,
    "height": 63,
    "weight": 130,
    "smoke": 0
  }'
```

**Response:**
```json
{
  "predicted_class": 1,
  "predicted_label": "স্বাভাবিক",
  "predicted_range": "88 – 141 oz  (2.5 – 4 kg)",
  "bmi": 23.01,
  "probabilities": {
    "low": 0.0821,
    "normal": 0.8134,
    "high": 0.1045
  }
}
```

---

## ক্লাস সংজ্ঞা

| ক্লাস | মান | বাস্তব ওজন |
|-------|-----|------------|
| 0 — কম ওজন | < 88 oz | < 2.5 kg |
| 1 — স্বাভাবিক | 88–141 oz | 2.5–4 kg |
| 2 — বেশি ওজন | > 141 oz | > 4 kg |
