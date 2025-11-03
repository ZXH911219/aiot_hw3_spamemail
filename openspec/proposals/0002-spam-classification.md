---
title: "0002 - Spam email classification (baseline + roadmap)"
authors: ["user <user@local>"]
status: draft
created: 2025-10-31
tags: [ml, spam, classification, baseline]
---

# Summary

This proposal describes a phased plan to build a spam email / SMS classification system using machine learning. The initial baseline (Phase 1) will implement a straightforward logistic regression classifier trained on a public SMS spam dataset. The dataset for Phase 1 is available at:

https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

Phases after the baseline are left intentionally empty in this proposal and will be defined later as follow-up proposals (Phase 2, Phase 3, ...).

# Motivation

Spam detection is a common and important classification problem with both practical and educational value. Starting with a simple, explainable baseline (logistic regression) will let us validate data pipelines, feature extraction, and evaluation metrics quickly before investing in more complex models.

# Detailed Design (Phase 1: Baseline)

Goal: Build a functional, documented baseline classifier using logistic regression and produce reproducible training and evaluation artifacts.

Data
- Source: the CSV above (public dataset from Packt's repo).
- Expected format: rows of <label>,<message> or similar (we will confirm by inspecting the CSV at implementation time). If the file lacks headers we will add them during ingest.

Preprocessing
- Load CSV into a pandas (or Node-friendly CSV) pipeline.
- Normalize text: lowercase, strip punctuation, optional basic tokenization.
- Remove or flag empty messages.

Feature Extraction
- Use TF-IDF vectorization (scikit-learn in Python or an equivalent JS library).
- Optionally experiment with n-grams (1-2) for baseline.

Model
- Baseline model: Logistic Regression (solver configurable).
- Note: you mentioned SVM in your plan; SVM is an alternative we can add as a Phase 2 experiment. For now we will implement Logistic Regression as the primary baseline per your stated goal.

Training
- Train/test split: 80/20 (or stratified k-fold CV, k=5 for robustness).
- Evaluate using precision, recall, F1-score, and ROC-AUC (where applicable).

Deliverables for Phase 1
- `data/` downloader script that fetches the CSV from the URL and saves a canonical copy in `data/raw/` with checksum.
- `notebooks/` or `src/` training script that performs preprocessing, trains the logistic regression baseline and emits a model file (e.g., `models/baseline-logreg.pkl` or a JSON artifact for JS-based models).
- Evaluation report: `reports/phase1-eval.md` containing confusion matrix and metrics.
- Minimal unit tests for data loader and preprocessing pipeline.

# Roadmap: Phases (placeholders)

- Phase 1 - baseline (logistic regression): implement, evaluate, and document. (Detailed above.)
- Phase 2 - [empty placeholder]
- Phase 3 - [empty placeholder]
- Phase 4 - [empty placeholder]

These phases will be filled with concrete experiments such as SVM/ensemble models, deep learning approaches, feature engineering (word embeddings), or deployment/online-scoring pipelines as the project progresses.

# Testing Plan

- Unit tests for data loading (valid CSV, missing values), for preprocessing functions (tokenization, normalization), and for model training entrypoints (sanity checks on outputs).
- Integration test: run the full training pipeline on a small sample and assert the pipeline completes and writes expected artifacts.
- Evaluation checks: ensure that the baseline achieves non-trivial performance (e.g., F1 > 0.6) on the test split; this threshold is provisional and will be adjusted after initial runs.

# Rollout & Rollback

- Rollout: Phase 1 is a non-production baseline for evaluation. If the model will later be used in production, we will add staging deployment steps and monitoring.
- Rollback: For any deployment, rollback to the previous model artifact and disable the new scoring endpoint.

# Acceptance Criteria

- A reproducible training pipeline that downloads the dataset, trains a logistic regression model, and produces an evaluation report.
- Scripts/tests and documentation present in the repository under `data/`, `src/` or `notebooks/`, and `reports/`.

# Estimated Effort

- Phase 1: 1-3 days (download dataset, preprocess, train baseline, write report and tests).

# Notes & Assumptions

- Assumption: development will use Python (pandas, scikit-learn) by default. If you prefer JavaScript/Node for training, I can adapt the scripts to use `ml.js` or another JS ML stack.
- The CSV link is external; for reproducibility the pipeline will vendor a copy into `data/raw/`.
- If you want SVM as the baseline instead of logistic regression, tell me and I will update the proposal and scripts accordingly.

## Proposal management automation (merged from 0001)

To keep proposal files consistent and to automate changelog generation, we'll apply a small OpenSpec automation pipeline. This work was originally proposed in `openspec/proposals/0001-automated-proposal-validation.md` and is merged here.

Summary

- Add automated validation for OpenSpec change proposals and a CI-driven changelog generation step that compiles accepted proposals into a release changelog.

This automation will include:

- Proposal linting on PRs to validate front-matter and required sections.
- A merge-time job to gather `status: accepted` proposals and append their summaries to `CHANGELOG.md` or a release draft.

Detailed design (high-level)

1. Proposal Linting (PR-level)
	- A CI workflow (e.g., GitHub Actions) will run a script to validate proposal front-matter fields: title, authors, status, created, and mandatory body sections (Summary, Motivation, Detailed Design, Testing Plan, Rollout & Rollback, Acceptance Criteria). The check will report missing fields.

2. Proposal Acceptance & Changelog Generation (merge-level)
	- A CI job will pick proposals with `status: accepted` since the last release and generate/update `CHANGELOG.md` with grouped entries. Optionally create a draft release body.

3. Implementation details
	- Place proposal files under `openspec/proposals/`.
	- Add lightweight scripts under `tools/openspec-tools/` (e.g., `validate-proposal.js`, `generate-changelog.js`).
	- Add `.github/workflows/openspec.yml` with `validate` (PR) and `generate-changelog` (push/merge) jobs.

Migration and testing notes

- Start with the linter as advisory; make it blocking after a grace period.
- Unit tests for validator and an integration test for the changelog generator are recommended.

This automation is optional for Phase 1 model work, but recommended to improve traceability as more proposals and phases are added.
