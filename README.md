# HW3 - Spam Classification (OpenSpec)

This workspace contains an OpenSpec-based project for building a spam classification pipeline. The `openspec/` folder holds project context and proposals. The primary active proposal is `openspec/proposals/0002-spam-classification.md`.

Phase 1 scaffold: basic scripts for downloading, preprocessing, training, and predicting are included under `scripts/`.

Quick start (Windows PowerShell):

```powershell
# create and activate a virtual environment (optional but recommended)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# download dataset (vendor a copy)
python scripts/download_data.py --url "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv" --out data/raw/sms_spam_no_header.csv

````markdown
# HW3 - Spam Classification (OpenSpec)

This workspace contains an OpenSpec-based project for building a spam classification pipeline. The `openspec/` folder holds project context and proposals. The primary active proposal is `openspec/proposals/0002-spam-classification.md`.

Phase 1 scaffold: basic scripts for downloading, preprocessing, training, and predicting are included under `scripts/`.

Quick start (Windows PowerShell):

```powershell
# create and activate a virtual environment (optional but recommended)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# download dataset (vendor a copy)
python scripts/download_data.py --url "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv" --out data/raw/sms_spam_no_header.csv

# preprocess
python scripts/preprocess_emails.py --input data/raw/sms_spam_no_header.csv --output data/processed/sms_spam_clean.csv --no-header --label-col-index 0 --text-col-index 1

# train baseline
python scripts/train_spam_classifier.py --input data/processed/sms_spam_clean.csv --out models

# predict a single text
python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash" --models models

# run tests
pytest -q
```

If you'd like, I can also scaffold visualization scripts and a Streamlit app like the example repo.


````

## Live demo

You can view the deployed Streamlit demo here:

[Live demo — Streamlit app](https://littlecar.streamlit.app/)
