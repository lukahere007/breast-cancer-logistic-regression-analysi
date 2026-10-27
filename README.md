# Breast Cancer Diagnosis with Logistic Regression (R)

![R](https://img.shields.io/badge/R-4.3-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **This is the R version.** The same analysis in Python is in
> [breast-cancer-analysis-python](https://github.com/lukahere007/breast-cancer-analysis-python).

Can a small, interpretable logistic regression on the ten *mean* cell-nucleus features of the Breast Cancer Wisconsin (Diagnostic) dataset tell malignant from benign fine-needle aspirates? **Yes, fairly well.** Texture, concavity and concave points give a test AUC of 0.966 and catch 37 of 42 malignant tumours in a held-out test set used only once. With 113 test cases, the uncertainty is real (AUC 95% CI 0.939–0.993).

**Read the full rendered analysis:** [`breast_cancer_logistic_regression.md`](breast_cancer_logistic_regression.md) (code, output and figures, viewable on GitHub).

## Results (held-out test set, n = 113, 42 malignant)

| Metric | Value |
|---|---|
| Accuracy | 0.912 |
| Sensitivity (malignant caught) | 0.881 (37 / 42) |
| Specificity (benign cleared) | 0.930 (66 / 71) |
| ROC AUC | 0.966 (DeLong 95% CI 0.939–0.993) |

Final model, fit on train + validation:

```
logit(p) = -14.18 + 0.378*texture_mean - 8.51*concavity_mean + 141*concave_points_mean
```

`concavity_mean` and `concave_points_mean` are correlated at r = 0.94. Once concave points is in the model, the concavity coefficient flips sign and its CI includes 1, so read the two together as "nucleus shape" rather than as separate effects.

![Model comparison on the validation set](figures/model-comparison-plot-1.png)

![ROC curve on the test set](figures/roc-test-1.png)

## Method

1. **Data:** [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), 569 aspirates (357 benign, 212 malignant). Only the 10 `*_mean` features are used. The file is committed as `data/wdbc.csv`.
2. **Split:** stratified 60 / 20 / 20 with `caret::createDataPartition` (`set.seed(123)`).
3. **Screening (train only):** fit a full 10-feature `glm` and keep predictors with p < 0.05 as the base set. That gives texture and concavity. The VIFs reach about 1,300 (radius) and 940 (perimeter), so this screen is unstable.
4. **Selection (validation):** try the base set plus each of the other eight features. Fit each candidate on train and score it on validation by log loss, AUC and accuracy. Choose the lowest validation log loss, treating models within 0.01 as tied and preferring fewer features. Adding `concave_points_mean` wins clearly (log loss 0.160 vs 0.269 for the next best).
5. **Final evaluation (test, used once):** refit on train + validation and report the confusion matrix (malignant as the positive class), sensitivity, specificity, and AUC with a DeLong CI.

## What changed from the first version

- **Models were fit and scored on the same validation rows.** The earlier "model comparison" fitted every candidate on the 113-row validation set and then measured accuracy and AUC on those same rows, which rewards overfitting. The final model was also fit on the validation set only. Now candidates are fit on train, scored on validation, and the final model is refit on train + validation.
- **Sensitivity and specificity were swapped.** `caret::confusionMatrix` treats the first factor level (`B`) as positive by default, so the reported "sensitivity 92.96%" was really specificity. With `positive = "M"`, the original model's sensitivity was 0.857 and its specificity 0.930. Its AUC was 0.957 and accuracy 0.903 (reproduced here before the fix).
- **The code did not run from a clean clone.** It used a hardcoded `C:/Users/...` path and needed `ggstatsplot`/`ggpmisc`. Those boxplots are now plain `ggplot2` and drawn on the training set only.
- **The plot annotation hardcoded coefficients.** The equation is now computed from the fitted model.

The RPubs report ([rpubs.com/lukahere007/breast-cancer-logistic-regression](https://rpubs.com/lukahere007/breast-cancer-logistic-regression)) shows the earlier version and has not been updated yet.

## Run it

Requires R (4.3 tested) with the packages: `rmarkdown`, `knitr`, `dplyr`, `tidyr`, `ggplot2`, `caret`, `pROC`, `car`.

```r
install.packages(c("rmarkdown", "knitr", "dplyr", "tidyr", "ggplot2", "caret", "pROC", "car"))
rmarkdown::render("breast_cancer_logistic_regression.Rmd")                            # -> .md + figures/
rmarkdown::render("breast_cancer_logistic_regression.Rmd", output_format = "html_document")  # -> .html
```

Run from the repository root so that `data/wdbc.csv` resolves.

## Repository layout

```
├── breast_cancer_logistic_regression.Rmd   # the analysis
├── breast_cancer_logistic_regression.md    # rendered output (GitHub-viewable)
├── figures/                                # plots written by the render
└── data/wdbc.csv                           # UCI WDBC data (569 rows)
```

## Limitations

- **Small test set.** One misclassified tumour moves sensitivity by about 2 points.
- **Unstable feature screening.** p-value screening on collinear features depends on the split. The Python version, with a different random split, screens in texture + area + concave points (test AUC 0.977). Both final models land in a similar performance range.
- **The 0.5 threshold is arbitrary.** A missed malignancy costs more than a false alarm, so a real screening tool would set the threshold for a target sensitivity.
- **Only the `_mean` features are used.** The `_se` and `_worst` features would likely help.
- **Not a clinical tool.** The data come from one centre and the model has not been externally validated.

## Author

Luke Wamalwa · [LinkedIn](https://www.linkedin.com/in/luke-wamalwa-839624292)
