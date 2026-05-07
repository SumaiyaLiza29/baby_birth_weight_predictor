Baby Birth Weight Prediction — Full Project
Folder Structure
baby_bwt_predictor/
├── data/
│   └── final_continuous_babies_data.csv   ← Place your data here
├── models/                                ← After training, .pkl files will be saved here
├── backend/
│   ├── main.py                            ← FastAPI server
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
├── train.py                               ← Model training script
└── README.md
Step 1 — Train the Model

To train the model, follow these steps:

Navigate to the project folder:

cd baby_bwt_predictor

Install required Python packages (only once):

pip install -r backend/requirements.txt
Place your data:
Place the final_continuous_babies_data.csv file in the data/ folder.

Train the model:

python train.py

This will train the models and save the following files in the models/ folder:

lgbm.pkl
xgb.pkl
gb.pkl
meta_model.pkl
scaler.pkl
Step 2 — Start the Backend

Navigate to the project folder if you're not already there:

cd baby_bwt_predictor

Start the FastAPI server:

uvicorn backend.main:app --reload --port 8000
Check if the server is running by visiting:
Health check: http://localhost:8000/health
API docs: http://localhost:8000/docs
Step 3 — Start the Frontend

Navigate to the frontend folder in a new terminal window:

cd baby_bwt_predictor/frontend

Install dependencies:

npm install

Run the frontend application:

npm run dev
Access the frontend in your browser:
Go to http://localhost:3000
 to see the React app in action.
Using the API (Directly via Command Line)

You can also interact with the FastAPI server directly using curl or similar tools.

Send a POST request to the /predict endpoint:

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

Response Example:

{
  "predicted_class": 1,
  "predicted_label": "Normal",
  "predicted_range": "88 – 141 oz  (2.5 – 4 kg)",
  "bmi": 23.01,
  "probabilities": {
    "low": 0.0821,
    "normal": 0.8134,
    "high": 0.1045
  }
}
Class Definitions

Here’s a breakdown of the classes and their meanings:

Class	Value	Actual Weight
0 — Low Weight	< 88 oz	< 2.5 kg
1 — Normal	88–141 oz	2.5–4 kg
2 — High Weight	> 141 oz	> 4 kg
Notes:
Data File: Make sure you place the final_continuous_babies_data.csv file in the data/ folder before training the model.
Dependencies: All required dependencies for both backend and frontend are included in the requirements.txt and package.json files, respectively.