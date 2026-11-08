# Generated experiment results

**Evaluation protocol:** one 80:20 stratified train/test split (random seed 42). All five models use the same split; preprocessing is fitted only on training rows.
**Positive class:** RB (ready biodegradable).

## Test-set comparison

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.8436 | 0.9105 |      0.7159 |   0.8873 | 0.7925 | 0.6792 |
| Decision Tree          |     0.8009 | 0.8375 |      0.6747 |   0.7887 | 0.7273 | 0.5764 |
| K-Nearest Neighbors    |     0.8483 | 0.9018 |      0.7532 |   0.8169 | 0.7838 | 0.6686 |
| Gaussian Naive Bayes   |     0.6682 | 0.8195 |      0.5038 |   0.9437 | 0.6569 | 0.4622 |
| Support Vector Machine |     0.8531 | 0.9181 |      0.7381 |   0.8732 | 0.8    | 0.6912 |

## Model observations

- **Logistic Regression:** AUC 0.910 indicates strong ranking separation; F1 is 0.792 and MCC is 0.679. On the positive RB class, recall is stronger than precision (precision 0.716, recall 0.887).
- **Decision Tree:** AUC 0.838 indicates moderate ranking separation; F1 is 0.727 and MCC is 0.576. On the positive RB class, recall is stronger than precision (precision 0.675, recall 0.789).
- **K-Nearest Neighbors:** AUC 0.902 indicates strong ranking separation; F1 is 0.784 and MCC is 0.669. On the positive RB class, precision and recall are relatively balanced (precision 0.753, recall 0.817).
- **Gaussian Naive Bayes:** AUC 0.819 indicates moderate ranking separation; F1 is 0.657 and MCC is 0.462. On the positive RB class, recall is stronger than precision (precision 0.504, recall 0.944).
- **Support Vector Machine:** AUC 0.918 indicates strong ranking separation; F1 is 0.800 and MCC is 0.691. On the positive RB class, recall is stronger than precision (precision 0.738, recall 0.873).

## Overall winner

**Support Vector Machine** has the highest MCC (0.6912) on this held-out split. MCC is used as the primary tie-breaker because the UCI class counts are unequal; confirm the choice against precision/recall and the assignment's full table.

## Positive-class confusion matrices

### Logistic Regression

Rows = actual [NRB, RB], columns = predicted [NRB, RB].

`[[115, 25], [8, 63]]`

### Decision Tree

Rows = actual [NRB, RB], columns = predicted [NRB, RB].

`[[113, 27], [15, 56]]`

### K-Nearest Neighbors

Rows = actual [NRB, RB], columns = predicted [NRB, RB].

`[[121, 19], [13, 58]]`

### Gaussian Naive Bayes

Rows = actual [NRB, RB], columns = predicted [NRB, RB].

`[[74, 66], [4, 67]]`

### Support Vector Machine

Rows = actual [NRB, RB], columns = predicted [NRB, RB].

`[[118, 22], [9, 62]]`
