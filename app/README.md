# Streamlit Demo App

This folder contains a Streamlit demo app for the spam-classification project.

Features:
- Load a processed dataset (`data/processed/sms_spam_clean.csv`) or upload your own CSV
- Show class distribution and top tokens per class
- Load model artifacts (`models/vectorizer.joblib`, `models/model.joblib`, `models/labels.joblib`) and perform single-text prediction
- Run evaluation visuals (confusion matrix, ROC/PR where applicable)

Run locally:

```powershell
# (optional) create and activate venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Deploy to Streamlit Cloud:
1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and create a new app pointing to this repo, branch `main` (or your branch), and main file `app/streamlit_app.py`.
3. Ensure the `requirements.txt` includes `streamlit` and other dependencies.

If you want, I can scaffold a `Procfile` or GitHub Actions workflow to help with deployment.
