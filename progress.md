# Project Progress Log

**Project:** Flight Delay Prediction Using Machine Learning
**Duration:** 5 days
**Outcome:** Complete ML pipeline — EDA, preprocessing, 5-model comparison, model persistence, Tkinter GUI

---

## Day 1 — Project Setup and Initial Dataset

### Goal
Set up the project environment and begin working with the first dataset.

### Work Done

- Created the project repository and folder structure
- Set up a Python virtual environment
- Installed dependencies: Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost, Joblib, Jupyter
- Selected the initial dataset: **2015 Flight Delays and Cancellations** (~5.8 million flights, 31 columns)
- Loaded the dataset and performed initial inspection: shape, data types, missing values
- Examined the column set:
  - Scheduling columns: `FL_DATE`, `DEP_TIME`, `CRS_DEP_TIME`, `ARR_TIME`, etc.
  - Operational columns: `TAXI_OUT`, `WHEELS_OFF`, `AIR_TIME`, `TAXI_IN`, `WHEELS_ON`, etc.
  - Delay breakdown columns: `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, etc.
  - Cancellation columns: `CANCELLED`, `CANCELLATION_CODE`
- Removed cancelled flights from the dataset
- Defined the binary target: `1` if arrival delay > 15 minutes, else `0`

### Status
Dataset loaded and understood. Target variable defined. Ready to begin EDA.

---

## Day 2 — Working with the Large Dataset and Decision to Switch

### Goal
Continue EDA, preprocess the first dataset, train models, and evaluate results.

### Work Done

**Class imbalance discovered:**

After removing cancelled flights and preparing the target, the class distribution was:

| Class | Count | Percentage |
|---|---:|---:|
| 0 (Not Delayed) | ~4.72M | 81.4% |
| 1 (Delayed) | ~1.08M | 18.6% |

This is a severe imbalance. A model that predicts "Not Delayed" for every single flight achieves approximately 81% accuracy without learning anything useful.

**Feature selection complexity:**

The 31-column dataset included several operational columns that are only known during or after the flight:

- `TAXI_OUT` — time spent taxiing before takeoff
- `WHEELS_OFF` — actual wheels-off time
- `AIR_TIME` — actual time in the air
- `TAXI_IN` — time spent taxiing after landing
- `ACTUAL_ELAPSED_TIME` — actual total flight time

Including these in a pre-departure prediction system would constitute data leakage. Removing them required careful column-by-column review.

**Models trained (first dataset):**

| Model | Accuracy | Recall (Delayed) | F1 (Delayed) |
|---|---:|---:|---:|
| Logistic Regression | 81.39% | ~0% | ~0% |
| Decision Tree | 82.07% | 9.67% | 16.72% |
| XGBoost | 81.83% | 3.66% | 6.98% |

**What these numbers actually mean:**

Logistic Regression essentially learned one rule: predict "Not Delayed" for everything. That single rule achieves 81.39% accuracy because the majority class is 81.4% of the data. The model never meaningfully detected a delayed flight.

Decision Tree and XGBoost barely improved recall. Despite the high accuracy numbers, none of the three models were practically useful for identifying the class we actually care about.

**Decision to switch datasets:**

The first dataset had two compounding problems:

1. Severe class imbalance made accuracy a misleading metric and made it difficult for models to learn the minority class without additional techniques like oversampling or class weights.
2. Many columns only become available during or after flight operations, which complicated the definition of a clean pre-departure prediction problem.

The project goal was:

```
Airline + airports + day + departure time + duration → Delay prediction
```

The first dataset did not map cleanly to this goal.

### Status
First dataset work concluded. Decision made to restart with a cleaner, more balanced dataset.

---

## Day 3 — New Dataset, Preprocessing, and Model Training

### Goal
Begin the project pipeline from scratch using the new dataset.

### New Dataset: Airlines.csv

| Property | Value |
|---|---|
| Records | 539,383 |
| Columns | 9 |
| Class balance | 55.46% Not Delayed / 44.54% Delayed |

The column set maps directly to pre-departure flight information:

```
Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length, Delay
```

No operational or post-departure columns are present. The prediction problem is well-defined.

### Work Done

**EDA:**
- Inspected dataset shape, data types, missing values, and target distribution
- Plotted: departure time distribution, delay rate by hour, delay rate by airline, delay rate by day of week, delay rate by flight duration, delay rate by departure airport (top 10)

**Feature engineering:**

`Time` is a cyclical variable. A flight at 23:50 and a flight at 00:10 are 20 minutes apart, but as a plain integer they are nearly 1440 units apart. Sine/cosine transformation fixes this:

```python
airlines["Time_sin"] = np.sin(2 * np.pi * airlines["Time"] / 1440)
airlines["Time_cos"] = np.cos(2 * np.pi * airlines["Time"] / 1440)
```

**Feature selection:**

Dropped `id`, `Flight`, `Time`, and the temporary `Hour` column.

Final feature set before encoding: 7 columns.

**Train-test split:** 80/20, stratified on target.

**Preprocessing pipeline:**

| Feature Type | Columns | Transformation |
|---|---|---|
| Categorical | `Airline`, `AirportFrom`, `AirportTo`, `DayOfWeek` | OneHotEncoder |
| Numerical | `Length`, `Time_sin`, `Time_cos` | StandardScaler |

After preprocessing: **614 features** per record.

**Models trained:**
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine (LinearSVC)

All five models trained on the same `X_train_processed` and `y_train`.

### Status
All five models trained. Ready for evaluation and comparison.

---

## Day 4 — Model Evaluation, Comparison, Model Saving, and Prediction Workflow

### Goal
Evaluate all five models, generate visualizations, save models, and build the prediction workflow.

### Work Done

**Evaluation metrics used:** Accuracy, Precision, Recall, F1-score, ROC-AUC

**Results:**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 65.04% | 63.68% | 50.05% | 56.05% | 69.70% |
| Decision Tree | 60.76% | 57.22% | 47.20% | 51.73% | 61.30% |
| Random Forest | 61.53% | 57.11% | 54.74% | 55.90% | 65.18% |
| KNN | 63.01% | 59.10% | 55.10% | 57.03% | 66.16% |
| SVM | 65.03% | 63.90% | 49.42% | 55.73% | 69.68% |

**Key observations:**
- Logistic Regression and SVM achieved the highest accuracy (~65%) and ROC-AUC (~69.7%)
- KNN achieved the best F1-score (57.03%), indicating the best precision-recall balance
- Decision Tree performed the weakest across all metrics
- The more balanced dataset meant no model could "cheat" by predicting the majority class

**Contrast with Day 2 results:**

The first dataset gave Logistic Regression 81.39% accuracy with ~0% recall for delayed flights. The second dataset gives Logistic Regression 65.04% accuracy with 50.05% recall. The second number looks lower, but the model is now actually detecting delayed flights rather than ignoring them.

**Plots generated:**
- Confusion matrix (Logistic Regression)
- ROC curve (Logistic Regression) — AUC: 0.697
- Model accuracy comparison bar chart
- Multi-metric model performance comparison chart

**Models saved:**

```python
joblib.dump(preprocessor,   "models/preprocessor.pkl")
joblib.dump(logistic_model, "models/logistic_regression.pkl")
joblib.dump(decision_tree,  "models/decision_tree.pkl")
joblib.dump(random_forest,  "models/random_forest.pkl")
joblib.dump(knn_model,      "models/knn.pkl")
joblib.dump(svm_model,      "models/svm.pkl")
```

**Prediction workflow built:**

Demonstrated end-to-end prediction on a new flight record using the saved preprocessor and Logistic Regression model, returning both the predicted class and the delay probability.

**Note on XGBoost:**

XGBoost was also trained as an additional experiment on the second dataset and achieved a ROC-AUC of approximately 0.7102, slightly higher than Logistic Regression. However, it was not included in the formal five-model comparison table and was not saved alongside the other five models. The five models above form the official comparison.

### Status
Full evaluation complete. All models and preprocessor saved. Prediction workflow tested. Ready to build GUI.

---

## Day 5 — GUI Development

### Goal
Build a user-facing application that wraps the trained model in a simple interface.

### Work Done

- Built a Tkinter GUI in `gui.py`
- GUI accepts: Airline, AirportFrom, AirportTo, DayOfWeek, departure time (converted to minutes), and flight duration
- On submission, the input is validated, assembled into a DataFrame, cyclical features are computed, and the saved preprocessor transforms the data
- The saved Logistic Regression model generates the prediction
- The interface displays: predicted class (Delayed / Not Delayed) and delay probability as a percentage
- Tested the GUI on several flight inputs and verified prediction output matches notebook output
- Added `cli.py` as a command-line alternative for predictions without the GUI
- Cleaned up the repository: updated `.gitignore` to exclude `Airlines.csv` and `models/` due to file size
- Finalized `README.md` and `progress.md`

### Status
Project complete.

---

## Summary

| Day | Focus | Outcome |
|---|---|---|
| Day 1 | Project setup, first dataset | Environment ready, dataset loaded |
| Day 2 | First dataset EDA and training | Models trained but results poor due to class imbalance; decision to switch datasets |
| Day 3 | New dataset, EDA, feature engineering, preprocessing, training | All five models trained on clean 614-feature matrix |
| Day 4 | Evaluation, comparison, saving, prediction workflow | Full results table, plots, saved models, prediction tested |
| Day 5 | GUI, CLI, cleanup | Tkinter GUI and CLI complete, repository finalized |

---

## Why the Dataset was Changed (for reference)

If asked why the project moved from the 2015 dataset to Airlines.csv, the complete technical explanation is:

The first dataset had severe class imbalance: approximately 81.4% of flights were not delayed. This caused models to achieve high accuracy by predominantly predicting the majority class. Logistic Regression reached 81.39% accuracy with near-zero recall for delayed flights. Decision Tree and XGBoost barely improved on this.

The dataset also contained several operational columns (taxi time, wheels-off time, actual elapsed time) that are only available during or after a flight. Building a clean pre-departure prediction system required removing these, which added substantial complexity.

The second dataset (Airlines.csv) has a 55/45 class split, contains only pre-departure information, and maps directly to the project goal: given what is known before a flight departs, predict whether it will be delayed.