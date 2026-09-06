# TECHTRACK 3.0 — Judge Demo Guide

This guide explains how to evaluate Team Voltra's submission.

## 1. Quick Verification (The Final Audit)
We have included a script that automatically audits our submission against the competition rules (no target leakage, outputs present, etc.).
Run this first to verify our integrity:
```bash
python scripts/final_audit.py
```
*Expected Output: "SUBMISSION READY: YES"*

## 2. Running the Full Pipeline
To train our model from scratch, execute:
```bash
python run_pipeline.py
```
This will:
1. Load and clean data (handling edge cases like 'Banana Boxes')
2. Engineer physical proxy features
3. Run the model arena (cross-validation on 13 models)
4. Tune hyperparameters and evaluate ensembles
5. Perform final evaluation and SHAP analysis
6. Run the physics sanity check

All results and figures are saved in the `outputs/` directory.

## 3. Interactive Web App Demo
We have provided an interactive Streamlit application to demonstrate our model's predictions on new vehicles.
```bash
streamlit run app/app.py
```

## 4. Reviewing the Code
We highly recommend reviewing our modular `src/` directory. 
- `src/preprocessing.py`: Demonstrates our clean pipeline approach.
- `src/feature_engineering.py`: Shows our physics-inspired feature creation.
- `src/explainability.py`: Shows our robust evaluation.

## 5. The Notebook
Our `notebooks/TECHTRACK3_Winning_Solution.ipynb` provides a narrative journey through our methodology and can be viewed directly in GitHub or Jupyter.
