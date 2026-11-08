# BITS Virtual Lab runbook and submission checklist

## One-time setup

```bash
python -m pip install -r requirements.txt
python train.py
```

Expected generated files:

- `test_data.csv` — held-out labelled test rows (all 41 features + target)
- `model/logistic_regression.joblib`
- `model/decision_tree.joblib`
- `model/k_nearest_neighbors.joblib`
- `model/gaussian_naive_bayes.joblib`
- `model/support_vector_machine.joblib`
- `model/feature_names.json` and `model/metadata.json`
- `results/model_metrics.csv` and `results/assignment_results.md`

## Launch and check

```bash
streamlit run app.py
```

Use the app's CSV uploader with `test_data.csv`. Confirm all six metrics render for all five models. Select each model and confirm its confusion matrix, ROC curve, and classification report change. Download the scored rows once as a smoke test.

## Final submission items from the assignment brief

- [ ] GitHub repository URL works and contains source, requirements, README, test CSV, and all five saved model artefacts.
- [ ] Streamlit Community Cloud URL opens the interactive app.
- [ ] One genuine screenshot shows execution inside BITS Virtual Lab.
- [ ] README includes the actual six-metric table and observations for all five models.
- [ ] Final single PDF contains, in order: GitHub link, live app link, BITS Lab screenshot, and full README content.
- [ ] Check the exact final PDF opens before uploading it.

Do not submit placeholder scores or links. The brief gives the deadline as **8 November 2026, 23:59**.
