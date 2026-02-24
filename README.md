# Titanic - Binary Classification

**Author:** Yassine KHERBOUCHE (@yassinemimis)  
**Task:** Binary Classification — Survival Prediction  
**Dataset:** Titanic (Kaggle) — 891 samples, 12 features  
**License:** MIT  

## Model
- **Algorithm:** Random Forest Classifier
- **Best F1:** ~0.82
- **CV (5-fold) F1:** ~0.81

## Run
```bash
pip install -e .
jupyter notebook notebooks/titanic_classification.ipynb
```

## Structure
```
├── data/          # train.csv, test.csv, saved model
├── notebooks/     # main notebook
├── src/           # model.py
└── pyproject.toml
```
