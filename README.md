# EcoTrace — QSAR Biodegradation Classification Lab

> **Learning project scaffold for the BITS M.Tech (AIML) Machine Learning Assignment 2.**
> Run the training workflow yourself in BITS Virtual Lab, inspect the generated results,
> customise the app, and write your own interpretation before submitting.

## a. Problem statement

Chemical biodegradability screening helps prioritise compounds for further environmental assessment. This project compares five classical supervised classifiers that predict whether a chemical is **ready biodegradable (RB)** or **not readily biodegradable (NRB)** from molecular descriptors. The app evaluates labelled, unseen test rows; it does not retrain a model when a user uploads a CSV.

This is a benchmark/teaching exercise, not a replacement for experimental testing or regulatory decisions.

## b. Dataset description

**Selected dataset:** [UCI QSAR biodegradation, ID 254](https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation), DOI [10.24432/C5H60M](https://doi.org/10.24432/C5H60M).

The repository describes 1,055 chemicals represented by 41 integer/continuous molecular descriptors and two experimental classes. It reports 356 ready-biodegradable and 699 not-ready-biodegradable chemicals, and no missing values. It therefore exceeds the assignment minimum of 500 records and 12 predictor features. The dataset is more specialised than common classroom sets such as Iris or Titanic, which gives the project a distinctive domain context. Its licence is CC BY 4.0; retain the attribution when sharing the data.

**Target convention in this project:** `RB` = ready biodegradable (positive class); `NRB` = not readily biodegradable. The generated CSV uses the target column `biodeg_class`.

**Source citation:** Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., & Consonni, V. (2013). *QSAR biodegradation* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5H60M.

## c. GitHub Repository Link

**Add your own public repository URL here after creating and pushing your project:** `PASTE_YOUR_GITHUB_REPOSITORY_URL`

Do not use another student's repository or claim this scaffold as independently authored work. Preserve meaningful commits showing your own data checks, experiments, UI changes, and interpretation.

## d. Models used

The following five required classifiers are fitted on the same stratified training split:

1. Logistic Regression — standardised descriptors; balanced class weights.
2. Decision Tree — bounded depth and minimum leaf size to limit overfitting; balanced class weights.
3. K-Nearest Neighbors — standardised descriptors; distance-weighted neighbours.
4. Gaussian Naive Bayes — standardised descriptors; Gaussian likelihood assumption.
5. Support Vector Machine — RBF kernel, standardised descriptors, probability estimates, balanced class weights.

**Evaluation protocol:** stratified 80:20 train/test split with `random_state=42`. Preprocessing is fitted inside each model pipeline using training rows only. The positive class for precision, recall, F1, and ROC-AUC is `RB`. MCC is also reported. This split is a reproducible baseline, not a substitute for repeated cross-validation.

### Comparison table

Run `python train.py`. It creates `results/model_metrics.csv` and `results/assignment_results.md` with the actual held-out values, and automatically replaces the placeholder rows in this README with the measured values:

| ML model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8436 | 0.9105 | 0.7159 | 0.8873 | 0.7925 | 0.6792 |
| Decision Tree | 0.8009 | 0.8375 | 0.6747 | 0.7887 | 0.7273 | 0.5764 |
| K-Nearest Neighbors | 0.8483 | 0.9018 | 0.7532 | 0.8169 | 0.7838 | 0.6686 |
| Gaussian Naive Bayes | 0.6682 | 0.8195 | 0.5038 | 0.9437 | 0.6569 | 0.4622 |
| Support Vector Machine | 0.8531 | 0.9181 | 0.7381 | 0.8732 | 0.8000 | 0.6912 |

### Observations and overall winner

Run-specific observations from the held-out test split:

- **Logistic Regression:** AUC 0.910 indicates strong ranking separation; F1 is 0.792 and MCC is 0.679. On the positive RB class, recall is stronger than precision (precision 0.716, recall 0.887).
- **Decision Tree:** AUC 0.838 indicates moderate ranking separation; F1 is 0.727 and MCC is 0.576. On the positive RB class, recall is stronger than precision (precision 0.675, recall 0.789).
- **K-Nearest Neighbors:** AUC 0.902 indicates strong ranking separation; F1 is 0.784 and MCC is 0.669. On the positive RB class, precision and recall are relatively balanced (precision 0.753, recall 0.817).
- **Gaussian Naive Bayes:** AUC 0.819 indicates moderate ranking separation; F1 is 0.657 and MCC is 0.462. On the positive RB class, recall is stronger than precision (precision 0.504, recall 0.944).
- **Support Vector Machine:** AUC 0.918 indicates strong ranking separation; F1 is 0.800 and MCC is 0.691. On the positive RB class, recall is stronger than precision (precision 0.738, recall 0.873).

**Overall winner:** Support Vector Machine leads by MCC (0.6912); F1 and AUC are used as tie-breakers. Review the confusion matrix and class-wise precision/recall before interpreting this as the most useful model.
## Project structure

```text
ecotrace_ml_assignment/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── test_data.csv                 # generated by train.py; commit this file
├── model/
│   ├── *.joblib                  # generated saved estimators; commit these files
│   ├── feature_names.json        # generated by train.py
│   └── metadata.json             # generated by train.py
├── results/
│   ├── model_metrics.csv         # generated by train.py
│   └── assignment_results.md     # generated by train.py
└── data/                         # generated training split files; do not need to upload
```

## Reproduce the experiment in BITS Virtual Lab

1. Create/open your BITS Virtual Lab environment and place this project folder there.
2. Open a terminal in the project folder.
3. Install dependencies: `python -m pip install -r requirements.txt`.
4. Fetch the public UCI data and train all models: `python train.py`.
5. Verify that `test_data.csv`, five `.joblib` files, `results/model_metrics.csv`, and `results/assignment_results.md` were created.
6. Inspect the metrics and class-wise confusion matrices. Add your own observations to the report.
7. Run the UI: `streamlit run app.py`.
8. In the app, upload `test_data.csv`; switch between all five models and inspect the comparison, confusion matrix, ROC curve, and classification report.
9. Capture the required screenshot **inside BITS Virtual Lab** showing your own execution. Do not fabricate or substitute a local screenshot.

## Streamlit Community Cloud deployment

1. Push the complete project to a GitHub repository. Include `app.py`, `train.py`, `requirements.txt`, `README.md`, `test_data.csv`, and the generated `model/` files.
2. Open [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Create a new app, select your repository and branch, set the main file path to `app.py`, and deploy.
4. Open the public app URL in a private/incognito browser window and test the upload using `test_data.csv`.
5. Paste your real repository and live app URLs into the final submission document.

**Important:** Streamlit Cloud does not run `train.py` automatically. The saved `.joblib` artefacts and `test_data.csv` must already be committed to the repository for the app to start without a training step.

## Streamlit app features

- CSV upload for labelled test data; the bundled test CSV is used on first load.
- Dropdown to select one of the five saved classifiers.
- Side-by-side six-metric comparison across all models.
- Selected-model metrics, confusion matrix, ROC curve, and classification report.
- Downloadable row-level prediction audit.
- No training or fitting is performed in `app.py`.

## Data schema

The file generated by `train.py` contains 41 numeric UCI descriptor columns and one target column named `biodeg_class`. Target labels must be `RB` or `NRB`. Keep all descriptor column names unchanged.

## What you should personalise before submission

- Change the colour palette, heading, layout, and explanatory copy to match your own design decisions.
- Add a short paragraph explaining the descriptor families and why scaling affects distance- and margin-based models.
- Discuss the false-negative/false-positive trade-off for RB, using your measured confusion matrices.
- Add the GitHub URL, live Streamlit URL, and a genuine BITS Lab screenshot to your final PDF.
- Keep source attribution and explain any design/code assistance you used in accordance with your institution's rules.
