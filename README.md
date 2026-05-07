
```markdown
# 👶 Baby Birth Weight Predictor

An end-to-end Machine Learning application featuring a **FastAPI** backend, a **React (Vite)** frontend, and an **Ensemble Model** (XGBoost, LightGBM, Gradient Boosting) to predict baby birth weight categories.

---

## 📂 Project Structure
```text
baby_bwt_predictor/
├── data/           # Place your CSV dataset here
├── models/         # Trained .pkl files and scalers will be saved here
├── backend/        # FastAPI server logic and requirements
├── frontend/       # React + Tailwind UI components
├── train.py        # Model training and pipeline script
└── README.md       # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Model Training
You must train the model before running the application to generate the necessary prediction files.

1. **Prepare Data:**  
   Place your `final_continuous_babies_data.csv` inside the `data/` folder.

2. **Install Dependencies:**  
   Run the following command to install required Python libraries:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Train the Model:**  
   Run the training script:
   
```bash
   python train.py
   ```
   *This will generate `.pkl` files and `scaler.pkl` in the `models/` directory.*

---

### 2️⃣ Backend Setup (FastAPI)
Once the models are trained, start the API server to handle requests:
```bash
# Navigate to the project root
uvicorn backend.main:app --reload --port 8000
```
* **Health Check:** `http://localhost:8000/health`
* **Interactive Docs:** Access the API documentation at `http://localhost:8000/docs`

---

### 3️⃣ Frontend Setup (React + Vite)
In a new terminal window, start the user interface:
```bash
cd frontend
npm install
npm run dev
```
* **URL:** Open `http://localhost:3000` in your browser.

---

## 📊 Class Definitions
The model categorizes the birth weight into three specific groups:

| Class | Label | Weight (Ounces) | Weight (Kilograms) |
| :--- | :--- | :--- | :--- |
| **0** | **Low Weight** | < 88 oz | < 2.5 kg |
| **1** | **Normal** | 88 – 141 oz | 2.5 – 4 kg |
| **2** | **High Weight** | > 141 oz | > 4 kg |

---

## 🛠 Tech Stack
* **Frontend:** React.js, Vite, Tailwind CSS
* **Backend:** FastAPI, Uvicorn
* **ML Models:** Scikit-Learn, XGBoost, LightGBM
* **Data Processing:** Pandas, NumPy

---
> **Note:** Ensure that your CSV file name in the `data/` folder matches the path defined in `train.py`.