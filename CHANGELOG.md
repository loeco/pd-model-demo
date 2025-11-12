# Changelog
## v0.1.0
- Base model: Logistic Regression
- Features: age, income, utilization
- AUC=0.74, KS=0.31

## v0.2.0
- Aggiunta feature: delinquency_count
- Re-training e tuning C
- AUC=0.77 (+0.03), KS=0.34

## v0.3.0
- Calibrazione isotonic
- Stability check: PSI<0.1
- AUC=0.77, Brier migliorato
