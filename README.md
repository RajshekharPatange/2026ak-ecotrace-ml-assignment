# EcoTrace – Intelligent Chemical Biodegradability Classification

**BITS M.Tech (AIML) | Machine Learning Assignment – 2**

**Project:** EcoTrace – QSAR Biodegradation Classification Lab

**Live Application:** https://ml2026ak05052rajp.streamlit.app/

**GitHub Repository:** https://github.com/RajshekharPatange/2026ak-ecotrace-ml-assignment

---

## A. Problem Statement

Chemical biodegradability assessment is important for understanding the potential environmental impact of chemical compounds.

The objective of this project is to develop a machine learning-based classification system that predicts whether a chemical compound is readily biodegradable (RB) or not readily biodegradable (NRB), using its molecular descriptor features.

Five supervised machine learning classification algorithms are implemented and evaluated on the same dataset.

The project also provides an interactive Streamlit application that allows users to upload labelled test data, select different classifiers, compare evaluation metrics, and inspect model performance through confusion matrices and classification reports.

The models are trained offline. The deployed application loads saved models and evaluates uploaded test data without retraining.

## B. Dataset Description

**Dataset Name:** QSAR Biodegradation Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation

**DOI:** https://doi.org/10.24432/C5H60M

The QSAR Biodegradation dataset contains chemical compounds represented by molecular descriptor features.

| Dataset characteristic | Description |
|---|---|
| Total instances | 1,055 |
| Input features | 41 |
| Classification type | Binary classification |
| Target variable | biodeg_class |
| Positive class | RB – Ready Biodegradable |
| Negative class | NRB – Not Readily Biodegradable |
| Readily biodegradable samples | 356 |
| Not readily biodegradable samples | 699 |
| Missing values | None reported in the source dataset |

The dataset satisfies the assignment requirements of at least 500 instances and 12 input features.

### Train-Test Split

The dataset was divided using stratified sampling:

- Training data: 80% (844 records)
- Testing data: 20% (211 records)
- Random state: 42

Stratification was used to maintain class proportions across training and testing data.

Feature scaling was incorporated inside the relevant model pipelines to prevent information leakage from the test set.

## C. GitHub Repository Link

**Repository:** https://github.com/RajshekharPatange/2026ak-ecotrace-ml-assignment

The repository contains:

- Complete Python source code.
- Model training and evaluation script.
- Streamlit application.
- Requirements file.
- Saved trained models.
- Test dataset in CSV format.
- Dataset attribution and documentation.
- Model evaluation results.

## D. Models Used and Evaluation

Five machine learning classification algorithms were implemented on the same dataset.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a class using a logistic function.

Configuration:
- StandardScaler preprocessing.
- Balanced class weights.
- Regularisation parameter C = 1.0.
- Maximum iterations = 3000.

### 2. Decision Tree Classifier

Decision Tree is a tree-based classification algorithm that uses feature-based decisions to classify chemical compounds.

Configuration:
- Gini impurity criterion.
- Maximum depth = 5.
- Minimum samples per leaf = 4.
- Balanced class weights.

### 3. K-Nearest Neighbors (KNN)

KNN classifies a compound using the nearest training samples in feature space.

Configuration:
- Number of neighbors = 7.
- Distance-based weighting.
- Euclidean distance.
- StandardScaler preprocessing.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem and the assumption of conditional independence between features.

Configuration:
- Gaussian likelihood.
- Variance smoothing = 1e-9.
- StandardScaler preprocessing.

### 5. Support Vector Machine (SVM)

Support Vector Machine identifies a decision boundary that separates the two classes. An RBF kernel is used to model nonlinear classification boundaries.

Configuration:
- RBF kernel.
- C = 1.5.
- Probability estimation enabled.
- Balanced class weights.
- StandardScaler preprocessing.

### Evaluation Metrics

The following six metrics were calculated for every model:

1. Accuracy
2. Area Under the ROC Curve (AUC)
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

Precision, recall and F1 are calculated considering RB as the positive class.

### Model Comparison Table

The following values are from the 211-row labelled holdout test evaluation displayed in the deployed EcoTrace application.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8436 | 0.9105 | 0.7159 | 0.8873 | 0.7925 | 0.6792 |
| Decision Tree | 0.8009 | 0.8375 | 0.6747 | 0.7887 | 0.7273 | 0.5764 |
| K-Nearest Neighbors | 0.8483 | 0.9018 | 0.7532 | 0.8169 | 0.7838 | 0.6686 |
| Gaussian Naive Bayes | 0.6682 | 0.8195 | 0.5038 | 0.9437 | 0.6569 | 0.4622 |
| Support Vector Machine | **0.8531** | **0.9181** | 0.7381 | 0.8732 | **0.8000** | **0.6912** |

### Model Performance Observations

**Logistic Regression**

Logistic Regression achieved an accuracy of 84.36% and an AUC of 0.9105. It also achieved a recall of 88.73%, demonstrating good identification of readily biodegradable compounds. Its performance indicates that a linear classification approach provides a strong baseline for this dataset.

**Decision Tree Classifier**

The Decision Tree achieved an accuracy of 80.09% and an AUC of 0.8375. Its performance was lower than Logistic Regression, KNN and SVM on this test set. The depth restriction provides a controlled model complexity, although the resulting classifier did not achieve the strongest predictive performance.

**K-Nearest Neighbors**

KNN achieved an accuracy of 84.83% and the highest precision of 75.32%. Its recall was 81.69%. The results indicate that distance-based classification provides competitive performance, particularly in identifying positive predictions accurately.

**Gaussian Naive Bayes**

Gaussian Naive Bayes achieved the highest recall of 94.37%, indicating that it identifies a large proportion of the readily biodegradable compounds. However, its precision was 50.38%, with an accuracy of 66.82%. The difference between recall and precision suggests a greater number of false-positive predictions compared with the stronger-performing classifiers.

**Support Vector Machine**

SVM achieved the highest accuracy of 85.31%, highest AUC of 0.9181, highest F1 score of 0.8000 and highest MCC of 0.6912. Its RBF kernel provides the ability to capture nonlinear decision boundaries. Considering the combined evaluation metrics, SVM demonstrated the strongest overall performance on the evaluated holdout dataset.

### Overall Winner

**Support Vector Machine (SVM)** is the overall winner for this dataset based on the evaluated results.

It achieved the best accuracy, AUC, F1 score and MCC among the five classifiers.

Although Gaussian Naive Bayes achieved higher recall and KNN achieved higher precision, SVM provided the strongest overall balance of classification performance.

This conclusion is limited to the evaluated holdout dataset and should not be interpreted as a guarantee of performance on every future dataset.

## E. Streamlit Web Application

**Live Application:** https://ml2026ak05052rajp.streamlit.app/

The EcoTrace application provides an interactive interface for comparing all five trained classification models.

### Implemented Features

1. CSV test-data upload functionality.
2. Automatic loading of the bundled holdout dataset.
3. Dropdown selection of the five trained classifiers.
4. Side-by-side comparison of six evaluation metrics.
5. Individual model performance inspection.
6. Confusion matrix visualization.
7. ROC curve visualization.
8. Classification report.
9. Test-row prediction audit.
10. No model retraining during application execution.

The application expects 41 molecular descriptor columns and a labelled target column named `biodeg_class`, with values RB or NRB.

## F. Project Structure

```text
2026ak-ecotrace-ml-assignment/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── test_data.csv
├── DATASET_ATTRIBUTION.txt
├── BITS_LAB_RUNBOOK.md
│
├── data/
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── feature_names.json
│   └── metadata.json
│
└── results/
    ├── model_metrics.csv
    └── assignment_results.md
```

## G. Installation and Execution

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and evaluate the five models:

```bash
python train.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

The training script saves the five trained models and generates the test data and evaluation results.

The Streamlit application loads the saved models and evaluates the labelled test data without performing model retraining.

## H. Deployment Details

The application was deployed using Streamlit Community Cloud.

**GitHub Repository:**

https://github.com/RajshekharPatange/2026ak-ecotrace-ml-assignment

**Live Streamlit Application:**

https://ml2026ak05052rajp.streamlit.app/

The deployed application provides access to the interactive EcoTrace classification interface.

## I. Conclusion

This project demonstrates an end-to-end machine learning classification workflow using a specialised chemical biodegradation dataset.

Five supervised learning algorithms were implemented, trained and evaluated using a common stratified holdout dataset. Their performances were compared using six standard classification evaluation metrics.

Support Vector Machine demonstrated the strongest overall performance, while Gaussian Naive Bayes offered the highest recall and KNN offered the highest precision.

The project also demonstrates model persistence, test-data evaluation, an interactive Streamlit interface, GitHub version control and cloud deployment.

The application is intended for educational and research demonstration purposes. Machine learning predictions should not replace laboratory biodegradation experiments or regulatory environmental assessments.

## J. Dataset Reference

Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., & Consonni, V. (2013). *QSAR biodegradation*. UCI Machine Learning Repository.

https://doi.org/10.24432/C5H60M

Dataset licence: CC BY 4.0.

---

**Project:** EcoTrace – QSAR Biodegradation Classification Lab  
**Programme:** M.Tech Artificial Intelligence and Machine Learning  
**Course:** Machine Learning  
**Assignment:** Assignment 2
