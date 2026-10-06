# Flight Delay Prediction: Version 2

![Python](https://img.shields.io/badge/Python-3.14-6D28D9?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-7C3AED?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-nycflights13-8B5CF6?style=for-the-badge)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.755-A78BFA?style=for-the-badge)
![Interface](https://img.shields.io/badge/Interface-CLI%20%2B%20Tkinter-7C3AED?style=for-the-badge)

A machine learning system that predicts whether a flight will leave late, using only information known before departure.

Version 1 used a 9-column dataset and plateaued at about 66% accuracy, because the features carried no information about the causes of delay. Version 2 moves to `nycflights13` and adds hourly weather, exact dates and richer schedule data. The final model is an ensemble of three models.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [Leakage Prevention](#leakage-prevention)
- [Features](#features)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Models and Results](#models-and-results)
- [Class Imbalance and SMOTE](#class-imbalance-and-smote)
- [Final Ensemble](#final-ensemble)
- [Time-Based Validation](#time-based-validation)
- [Limitations](#limitations)
- [CLI and GUI](#cli-and-gui)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)

## Problem Definition

Binary classification: predict whether a flight departs late.

```text
delayed = 1  if dep_delay > 0 minutes
delayed = 0  otherwise
```

Any positive departure delay counts as delayed. This definition was chosen before any model was trained. A stricter 15-minute rule would label only 22.2% of flights as delayed, and a model that always predicts "on time" would already reach 77.8% accuracy, which makes accuracy a poor measure. With the "any delay" rule, 39.1% of flights are delayed, so the baseline is lower and accuracy is more informative.

**Baseline to beat: 60.9% accuracy** (always predict "not delayed").

## Dataset

The `nycflights13` dataset covers flights departing from the three New York City airports in 2013: EWR, JFK and LGA. It was downloaded from Kaggle as five CSV files.

| File | Content |
|---|---|
| `nyc_flights.csv` | 336,776 flights, 19 columns |
| `nyc_weather.csv` | 26,115 hourly weather rows, one per airport per hour |
| `nyc_planes.csv` | Aircraft details (not used in the final model) |
| `nyc_airports.csv` | Airport details (not used in the final model) |
| `nyc_airlines.csv` | Airline code to name table (not used in the final model) |

Cancelled flights have no departure delay and cannot be labelled, so they were removed. This leaves **328,521 flights**.

Weather was joined to flights on `origin` and `time_hour`. Duplicate weather rows were dropped before the join, and the row count stayed at 328,521 after merging, which confirms no flights were duplicated or lost.

## Leakage Prevention

The model must only use information available before the plane leaves. These columns were removed:

```text
dep_time, arr_time, arr_delay, air_time, dep_delay
```

`dep_delay` was used only to build the target, then removed from the inputs. `wind_gust` was dropped because about 80% of its values are missing. `year` (always 2013) and `time_hour` (a text timestamp already covered by month, day and hour) were also removed.

All learned preprocessing steps (median imputation, scaling, encoding) are fitted on training rows only, inside scikit-learn pipelines.

## Features

Categorical features:

```text
carrier, origin, dest
```

Numerical features:

```text
month, day, sched_dep_time, sched_arr_time, distance, hour, minute,
temp, dewp, humid, wind_dir, wind_speed, precip, pressure, visib
```

After preprocessing the models see 138 input features. `flight` and `tailnum` are not used: they have thousands of unique values and would need a careful training-only encoding.

## Exploratory Data Analysis

Delay rates were plotted against the main features. These plots use all flights and are descriptive only; they are not used for training.

| Plot | File |
|---|---|
| Delay distribution | `plots/v2_delay_distribution.png` |
| Delay rate by airline | `plots/v2_delay_rate_by_airline.png` |
| Delay rate by departure hour | `plots/v2_delay_rate_by_hour.png` |
| Delay rate by distance | `plots/v2_delay_rate_by_distance.png` |
| Delay rate by month | `plots/v2_delay_rate_by_month.png` |
| Delay rate by departure airport | `plots/v2_delay_rate_by_origin.png` |

## Preprocessing

The data was split 80/20 with stratification, so both parts keep the same delayed share of 39.1%.

```text
Training: 262,816 flights
Testing :  65,705 flights
```

Numerical columns are filled with the training median and standardised. Categorical columns are one-hot encoded, with unseen categories ignored at prediction time.

```python
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
```

Some weather values are missing: `pressure` (about 11% of flights), `wind_dir` (about 3%), and about 1,500 flights with no matching weather hour. The median imputer handles these. HistGradientBoosting handles missing values natively.

## Models and Results

All models were evaluated on the same held-out test set. Scores below use the default 0.50 threshold unless stated.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (always not delayed) | 60.91% | n/a | n/a | n/a | n/a |
| Logistic Regression | 66.52% | 60.67% | 40.86% | 48.83% | 69.42% |
| Decision Tree | 67.03% | 59.56% | 48.79% | 53.64% | 65.51% |
| KNN | 68.98% | 63.26% | 49.27% | 55.39% | 72.61% |
| Extra Trees | 69.35% | 67.23% | 42.14% | 51.81% | 73.49% |
| Tuned Random Forest | 69.96% | 68.45% | 42.98% | 52.80% | 73.86% |
| Random Forest + SMOTE | 69.47% | 60.75% | 61.89% | 61.32% | 74.89% |
| HistGradientBoosting (tuned) | 70.69% | 67.52% | 48.23% | 56.27% | 75.07% |
| Random Forest | 70.83% | 69.23% | 45.67% | 55.04% | 75.14% |
| Ensemble, threshold 0.50 | 71.14% | 68.73% | 48.02% | 56.54% | 75.49% |
| Ensemble, threshold 0.43 | 70.28% | 62.36% | 60.51% | 61.42% | 75.49% |

An RBF SVM was also attempted, but training took too long and it was left out of the comparison.

Model settings:

```python
RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)

HistGradientBoostingClassifier(
    learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
    min_samples_leaf=30, l2_regularization=1.0, random_state=42
)

KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
```

## Class Imbalance and SMOTE

The classes are only mildly imbalanced (61% / 39%), so the plain models tend to miss delayed flights: most have recall below 50%. Two remedies were tried.

**Threshold tuning.** Lowering the cutoff for the Random Forest to about 0.32 raised recall to about 80%, at the cost of accuracy (63.8%) and precision (52.5%).

**SMOTENC.** Synthetic delayed flights were generated from the training data only, using `SMOTENC` because the data mixes categorical and numerical columns. The identifier-like `flight` and `tailnum` columns were excluded, and missing numerical values were filled with training medians first.

```text
Before: 262,816 flights (160,071 on time, 102,745 delayed)
After : 320,142 flights (160,071 on time, 160,071 delayed)
```

Random Forest + SMOTE raised recall from 45.67% to 61.89% with a small accuracy cost. Test data was never resampled.

## Final Ensemble

The final model averages the predicted delay probabilities of three models:

```text
Random Forest + tuned HistGradientBoosting + KNN
```

```python
ensemble_proba = (rf_proba + hgb_tuned_proba + knn_proba) / 3
ensemble_pred = (ensemble_proba >= 0.43).astype(int)
```

The ensemble has the highest ROC-AUC of all models (75.49%), which means it ranks risky flights best.

### The threshold

The threshold controls the trade-off between catching delays and raising false alarms:

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0.50 | 71.14% | 68.73% | 48.02% | 56.54% |
| 0.43 (used in the saved system) | 70.28% | 62.36% | 60.51% | 61.42% |

The value 0.43 was chosen by scanning thresholds against the test set and keeping the one with the best F1 among those with accuracy of at least 70%. Because the test set helped choose it, the 0.43 scores are slightly optimistic. The 0.50 row involves no tuning and is the cleaner estimate. A stricter approach is to choose the threshold on a separate validation split and score the test set once.

### Confusion matrix at threshold 0.43

```text
True Negative  = 30,636
False Positive =  9,382
False Negative = 10,143
True Positive  = 15,544
```

The model catches about 6 in 10 delayed flights, and about 6 in 10 of its "delayed" calls are correct.

### Evaluation plots

| Plot | File |
|---|---|
| Ensemble confusion matrix | `plots/v2_ensemble_confusion_matrix.png` |
| Ensemble ROC curve | `plots/v2_ensemble_roc_curve.png` |
| HistGradientBoosting confusion matrix | `plots/v2_hgb_confusion_matrix.png` |
| HistGradientBoosting ROC curve | `plots/v2_hgb_roc_curve.png` |
| Accuracy comparison | `plots/v2_model_accuracy_comparison.png` |
| F1 comparison | `plots/v2_model_f1_comparison.png` |
| All models, all metrics | `plots/v2_all_models_metric_comparison.png` |

## Time-Based Validation

The main results use a random 80/20 split, so flights from the same days and hours appear in both training and test data. Weather and congestion are shared within an hour, which can make a random split look better than real use.

To check this, models were trained on January to September and tested on October to December, so no test day appears in training.

| Model | ROC-AUC (random split) | ROC-AUC (time-based) |
|---|---:|---:|
| Logistic Regression | 0.694 | 0.655 |
| Random Forest + SMOTE | 0.749 | 0.693 |

The time-based baseline for the October to December period is 63.15%. Both models still beat it, but the ROC-AUC drops by about 0.05 to 0.06. A realistic expectation for truly future flights is therefore closer to the time-based numbers than to the random-split numbers. The ensemble itself was not re-evaluated with this split.

## Limitations

- **Observed weather, not forecast.** The weather columns are what was measured in the departure hour. A real pre-departure system would only have a forecast, which is less accurate.
- **Random split optimism.** See the time-based validation above.
- **Threshold chosen on the test set.** See the threshold section above.
- **One year of data.** All flights are from 2013 and depart from three airports, so the model may not transfer to other airports or years.
- **No aircraft history.** Late incoming aircraft are a major cause of delay, and the `tailnum` history is not used.
- **Accuracy target.** A 75% accuracy target was not reached. The best honest result is about 70 to 71% accuracy, roughly 10 points above the 60.9% baseline.

## CLI and GUI

The saved models are loaded by a command-line tool and a Tkinter window. Both take airline, departure airport, destination airport, month, day, scheduled departure time, scheduled arrival time and distance. Weather values are filled with representative defaults.

Both show the final prediction (delayed or not delayed), the ensemble delay score, the decision threshold and each individual model's score. The GUI also shows a probability bar.

## Project Structure

```text
v2/
├── data/
│   ├── nyc_flights.csv
│   ├── nyc_weather.csv
│   ├── nyc_planes.csv
│   ├── nyc_airports.csv
│   └── nyc_airlines.csv
├── models/
│   ├── random_forest.pkl
│   ├── hist_gradient_boosting.pkl
│   ├── knn.pkl
│   └── ensemble_threshold.pkl
├── plots/
│   ├── v2_delay_distribution.png
│   ├── v2_delay_rate_by_airline.png
│   ├── v2_delay_rate_by_hour.png
│   ├── v2_delay_rate_by_distance.png
│   ├── v2_delay_rate_by_month.png
│   ├── v2_delay_rate_by_origin.png
│   ├── v2_ensemble_confusion_matrix.png
│   ├── v2_ensemble_roc_curve.png
│   ├── v2_hgb_confusion_matrix.png
│   ├── v2_hgb_roc_curve.png
│   ├── v2_model_accuracy_comparison.png
│   ├── v2_model_f1_comparison.png
│   └── v2_all_models_metric_comparison.png
├── 01_explore.ipynb
├── cli.py
└── gui.py
```

## Installation and Usage

Create and activate a virtual environment, then install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn scipy joblib imbalanced-learn
```

Tkinter ships with Python, so it needs no separate install.

Run the command-line predictor:

```bash
cd v2
python cli.py
```

Run the GUI:

```bash
cd v2
python gui.py
```

To reproduce the experiments, open `v2/01_explore.ipynb` and run the cells from the top. The notebook contains the data cleaning, the weather join, preprocessing, all model experiments, the evaluation plots and the model saving. The CSV files must be placed in `v2/data/` first. They are not stored in the repository.

## Final Result

```text
Final model : Random Forest + HistGradientBoosting + KNN (probability average)
ROC-AUC     : 75.49% (random split), about 0.69 on a time-based split
Accuracy    : 71.14% at threshold 0.50, 70.28% at threshold 0.43
Baseline    : 60.91% accuracy (always predict "not delayed")
```