# NBA Player Salary Predictor

> A comparative regression study that asks a simple question — **can a machine predict how much an NBA player earns from their on-court stats?** — and ends up with a more interesting answer than I expected.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![ML](https://img.shields.io/badge/scikit--learn-%20XGBoost%20%7C%20Random%20Forest-orange)
![Status](https://img.shields.io/badge/status-complete-green)

---

## What it does

Given an NBA player's performance stats for the 2022-23 season (points, assists, rebounds, efficiency metrics, position, age, minutes played), this project predicts their salary — and compares **seven different regression techniques** to see which actually works best on real, noisy sports data.

The dataset is the [NBA Salaries Dataset from Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/nba-salaries), covering 2022-23 season performance and compensation.

## Why I built it

Originally I wanted to see if *polynomial regression* could capture the "superstar premium" — the idea that elite players earn exponentially more than average ones, not linearly. The hypothesis made intuitive sense. The data disagreed. That contradiction is what made the project interesting.

It turned into a broader study of how different regression families handle the same problem, and what happens when you push a model past the complexity it can actually support with the data you have.

## The approach

```
Raw NBA stats CSV
      ↓
Data cleaning → drop irrelevant columns, handle missing TS%, standardize Player names
      ↓
Feature engineering → one-hot encode Position, StandardScaler on numeric features
      ↓
EDA → scatterplots, correlation heatmap, log-transform for skewed salary distribution
      ↓
Model training → seven different regressors, same features, same split
      ↓
Evaluation → MAE, RMSE, R² across all models
      ↓
Residual & bias analysis → where does the model systematically fail?
```

## Models compared

| Model | Role in the study |
|---|---|
| **Linear Regression** | Baseline — interpretable and simple |
| **Polynomial Regression (deg 2 & 3)** | Testing the "superstar premium" hypothesis |
| **Lasso Regression** | Regularization + automatic feature selection |
| **Ridge Regression** | Regularization without shrinking features to zero |
| **Random Forest Regressor** | Ensemble, handles non-linearity natively |
| **SVR (RBF + Linear kernels)** | Kernel-based non-linear regression |
| **XGBoost Regressor** | Gradient boosting — usually the strongest baseline on tabular data |

## Key finding — the counterintuitive result

On the face of it, you'd expect polynomial regression to beat linear because salaries *feel* exponential. But the numbers said the opposite:

| Model | R² | MAE |
|---|---|---|
| **Linear Regression** | **0.66** | 0.49 |
| Polynomial Regression (deg=2) | **−0.06** | 0.71 |

A **negative R² means the model performs worse than just predicting the mean salary for everyone.** The polynomial model overfit the training data so badly it couldn't generalize.

**Why?** With ~20 base features and degree=2, you generate 210 polynomial terms. On a dataset of a few hundred players, that's a recipe for fitting noise instead of signal — the model memorizes rather than learns.

This taught me a lesson more valuable than the model itself: **complexity without enough data to support it makes things worse, not better.**

## The strongest models

After regularization and ensemble methods were added, the ranking became:

- **XGBoost** — best overall, handled non-linearity without overfitting
- **Random Forest** — close second, easier to interpret via feature importances
- **Linear Regression** — surprisingly competitive baseline
- **Lasso** — good feature selection, zeroed out most position indicators
- **Ridge** — similar to Linear but with smaller coefficients
- **SVR** — middle of the pack
- **Polynomial** — don't bother without regularization

## Feature insights

From the Random Forest feature importance plot and linear coefficients:

**Most predictive of salary:**
- `PTS` (points per game) — the strongest single predictor
- `VORP` (Value Over Replacement Player) — advanced stat that correlates heavily
- `MP` (minutes played) — proxy for how much a team trusts a player
- `Age` — veterans earn more, but with diminishing returns

**Surprisingly undervalued:**
- Defensive stats (`STL`, `BLK`, `TRB`) have positive coefficients but smaller magnitude. The NBA pays for scoring. Elite defenders who don't score tend to be underpaid by the market — a known bias the model picks up on.

## Tech stack

- **Python 3.10+**
- **pandas**, **NumPy** — data manipulation
- **scikit-learn** — preprocessing, Linear/Polynomial/Lasso/Ridge/SVR, metrics
- **XGBoost** — gradient boosting regressor
- **matplotlib**, **seaborn** — EDA and result visualization
- **scipy**, **statsmodels** — statistical summaries and z-score outlier detection

## Running it

```bash
git clone https://github.com/Iyimoga/NBA-Salary-Predictor.git
cd NBA-Salary-Predictor
pip install -r requirements.txt
jupyter notebook NBA.ipynb
```

You'll need the NBA salaries CSV from [the Kaggle dataset](https://www.kaggle.com/datasets/kaushiksuresh147/nba-salaries) placed in the project folder.

## What I learned

1. **More complexity ≠ better model.** Polynomial features looked theoretically right and performed practically wrong.
2. **R² can go negative.** It's not bounded between 0 and 1 — a negative R² is a flashing red light that your model is worse than useless.
3. **Regularization isn't optional.** Lasso and Ridge turned a failing polynomial idea into something workable.
4. **The best model is usually boring.** XGBoost and Random Forest beat everything without needing clever feature engineering — the classic "just try gradient boosting first" lesson holds up.
5. **Models inherit market biases.** The NBA undervalues defense; so does any model trained on its salary data. Worth remembering any time you deploy an ML system that predicts something people get paid for.

## About

Built by **[Iyimoga Joseph Nana](https://github.com/Iyimoga)** — CS student exploring applied ML on real-world tabular datasets.

📫 Reach out: Josephiyimoga05@gmail.com
