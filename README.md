# MLflow Experiment Tracking & Data Drift Analysis — Diabetes Progression

**Student:** Mayur Khamamkar (DS 03) · B.Tech Data Science · A.Y. 2026-27
**Course assessment:** Theory ISE-2 — MLflow Installation, Experiment Tracking and Data Drift Analysis

## What this does
1. Tracks a RandomForest regression experiment in **MLflow** (params, metrics, model, artifacts).
2. Checks for **data drift** and **prediction drift** between a younger patient cohort (reference) and an older cohort (current) in the diabetes dataset.
   It uses the **Kolmogorov–Smirnov test** and the **Population Stability Index (PSI)**.

## Dataset
`sklearn.datasets.load_diabetes` (Efron et al., 2004, *Least Angle Regression*). It has 442 patients and 10 baseline features.
The features are age, sex, BMI, blood pressure (bp) and six blood-serum measurements: tc, ldl, hdl, tch, ltg and glu.
The target is a quantitative measure of disease progression one year after baseline.

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_and_drift.py
mlflow ui --backend-store-uri sqlite:///mlflow.db    # open http://127.0.0.1:5000
```

## Drift setup
| | Rows | Rule |
|---|---|---|
| Reference | 222 | age ≤ 50 (younger cohort, model trained here) |
| Current | 220 | age > 50 (older cohort, model applied here) |

A feature is flagged as **drifted** if KS p-value < 0.05 **or** PSI > 0.2.
Age is the split key, so it is excluded from the verdict.
As a control, two random halves of the reference data are compared. Little or no drift is expected there.

## Results (run `55ecc2a2517b4d19b081d2b971362afc`)
| Column | Ref mean | Cur mean | KS p | PSI | Drift |
|---|---|---|---|---|---|
| sex | 1.39 | 1.55 | 0.0075 | 0.10 | Yes |
| bmi | 25.81 | 26.97 | <0.0001 | 0.44 | Yes |
| bp | 90.81 | 98.70 | <0.0001 | 0.55 | Yes |
| tc | 183.59 | 195.00 | 0.0003 | 0.20 | Yes |
| ldl | 111.80 | 119.29 | 0.0023 | 0.17 | Yes |
| hdl | 50.32 | 49.23 | 0.0177 | 0.20 | Yes |
| tch | 3.92 | 4.23 | 0.0001 | 0.15 | Yes |
| ltg | 4.54 | 4.75 | 0.0003 | 0.22 | Yes |
| glu | 88.64 | 94.03 | <0.0001 | 0.32 | Yes |
| **prediction** | 136.58 | 150.75 | 0.0008 | 0.18 | **Yes** |
| target (label) | 137.38 | 167.71 | 0.0026 | 0.18 | Yes |

* All 9 monitored features drifted. The strongest shifts are in bp, bmi and glu.
* Model predictions also drifted, so there is prediction drift.
* Model error **did not get worse**. Reference out-of-fold RMSE was 60.2 and MAE 47.9. On the current cohort, RMSE was 57.2 and MAE 46.5.
* The model also **under-predicts** progression for older patients. Mean prediction is 150.8, but the mean actual value is 167.7.
* Control check: 1 of 9 features was flagged between two random halves of the same data. That fits the expected false-positive rate of an OR'd test at α = 0.05.

## Repository layout
```
train_and_drift.py              # full pipeline (training + MLflow logging + drift tests + plots)
requirements.txt
reports/drift_report.csv        # per-feature KS / PSI table
reports/feature_distributions.png
reports/psi_bar.png
reports/summary.json
screenshots/mlflow_experiment.png
screenshots/mlflow_run_overview.png
mlflow.db, mlruns/              # MLflow tracking store + logged artifacts
```
