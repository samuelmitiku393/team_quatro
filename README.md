# Addis Ababa Ride Demand Forecasting Challenge
**Team Quatro** | Qiyas / IADE AI Training Program Hackathon

An end-to-end data science and forecasting system predicting hourly ride requests across 12 zones in Addis Ababa for November 1–14, 2025. The project integrates raw trip logs with hourly weather readings and a multi-zone city events calendar, adhering strictly to time-ordered chronological validation, leakage prevention, and automated data integrity checks.

---

## 👥 Team Quatro Members
- **Dagim Asnake** — Data Lead & Integration Pipeline
- **Modeling & Machine Learning Lead** — Model Architecture, Time-Series Splits & Evaluation
- **Analysis & Visualizations Lead** — Exploratory Data Analysis & 12-Figure Pack
- **Deployment Lead** — Streamlit Forecaster Demo Application

---

## 🏆 Final Model & Validation Performance
- **Selected Model:** LightGBM Regressor (`n_estimators=2000`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=20`, categorical zone handling).
- **Validation Holdout (October 2025, 8,690 zone-hours):**
  - **Mean Baseline:** RMSE = 30.16 | MAE = 20.79
  - **Seasonal-Naive Baseline:** RMSE = 17.16 | MAE = 7.53
  - **Final Model (Full Features):** **RMSE = 15.79** | **MAE = 6.76** (Tuned: 15.73)
  - **Rolling-Origin Validation (4 × 14-day folds):** **13.97 ± 2.58 RMSE**
- **Operational Metric:** RMSE corresponds to ~12.1 drivers/zone-hour and MAE to ~5.2 drivers/zone-hour (~20.3% of mean demand) at ~1.3 trips/driver/hour.

---

## 📁 Repository & Deliverables Layout

```text
team_quatro/
├── README.md                                 # Setup, run order, summary, scores, demo link (G2)
├── requirements.txt                          # Pinned dependency specifications (G3)
├── submission/
│   └── team_quatro_submission.csv            # 4,032 test predictions for Nov 1–14 (G4 / Scored)
├── data/
│   ├── raw/                                  # Original immutable data exports (5 CSVs)
│   └── processed/
│       ├── master_train.csv                  # Cleaned, integrated master training table (A8)
│       ├── master_test.csv                   # Cleaned, integrated test feature grid (A8)
│       └── data_dictionary_master.csv        # Comprehensive metadata dictionary (A8)
├── notebooks/
│   ├── 01_cleaning_and_integration.ipynb     # Deliverable A interactive pipeline
│   ├── 02_analysis_report.ipynb              # Deliverable B 14 analysis tasks
│   ├── 03_visualizations.ipynb               # Deliverable C 12-figure generation
│   └── 04_modeling_and_evaluation.ipynb      # Deliverable D modeling and validation
├── src/
│   ├── cleaning.py                           # Timestamp parsing, zone aliasing, deduplication
│   ├── integration.py                        # UTC->EAT conversion, weather joins, event windows
│   ├── features.py                           # Calendar, trend, weather, and event feature groups
│   ├── train.py                              # D1-D9 model training, ablation, tuning, export
│   ├── predict.py                            # Test inference and submission verification
│   └── visualizations.py                     # High-resolution (150 DPI) figure generator
├── models/
│   └── final_model.joblib                    # Serialized LightGBM model bundle (model + features)
├── figures/
│   ├── fig01_gaps_and_missingness.png ...    # 12 required figures (fig01 to fig12)
│   └── figure_captions.md                    # 2-sentence takeaways per figure (C)
├── reports/
│   ├── A_cleaning_and_integration.md         # Deliverable A log, join map, audit, checks (A1–A8)
│   ├── B_analysis_report.md                  # Deliverable B writeup for tasks B1.1–B4.3
│   └── D_model_evaluation.md                 # Deliverable D writeup for tasks D1–D9
├── app/
│   ├── app.py                                # Streamlit forecast demo web interface (E)
│   ├── requirements.txt                      # App dependencies
│   └── assets/                               # Bundled weather & event lookup tables
└── presentation/
    └── team_quatro_slides.pptx               # 5-slide PowerPoint widescreen presentation (F)
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
Clone the repository and install all dependencies:

```bash
git clone https://github.com/dagix7/team_quatro.git
cd team_quatro
pip install -r requirements.txt
```

### 2. Execution Order (Reproducing from Scratch)
Execute the modular pipeline in numerical order:

```bash
# Step 1: Run full modeling pipeline (D1-D9), generate model & export master tables
python src/train.py

# Step 2: Generate test predictions & create valid 4,032-row submission file
python src/predict.py

# Step 3: Generate the 12 Deliverable C figures at >=1200px (150 DPI)
python src/visualizations.py
```

### 3. Launching the Forecast Demo App (Deliverable E)
Run the Streamlit application with a single command from the project root:

```bash
streamlit run app/app.py
```
*(Or navigate into `app/` and run `streamlit run app.py`)*

The dashboard allows selecting any of the 12 Addis Ababa zones and any date strictly between **November 1 and November 14, 2025** to view hourly forecasts, peak hours, driver fleet estimates, and silent background weather/event lookups.
