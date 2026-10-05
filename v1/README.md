# Flight Delay Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-8b5cf6?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-7c3aed?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-6d28d9?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-5b21b6?style=for-the-badge)

A machine learning-based binary classification system that predicts whether a flight is likely to be delayed, built on 539,383 historical flight records. The project covers the complete ML workflow: exploratory data analysis, feature engineering, preprocessing, model training, evaluation, model persistence, and a Tkinter GUI.

---

## Abstract

Flight delays are a persistent challenge in the aviation industry. They affect passengers, airline operations, and airport scheduling. This project builds a binary classifier using historical flight data to predict delay status before departure.

The preprocessing pipeline applies cyclical transformation to departure time, One-Hot Encoding to categorical features, and StandardScaler to numerical features, producing a 614-feature matrix. Five classification algorithms are formally trained and evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC. The trained models and preprocessor are saved with Joblib and connected to a lightweight Tkinter GUI for interactive predictions.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Feature Engineering](#4-feature-engineering)
5. [Preprocessing](#5-preprocessing)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Model Persistence](#8-model-persistence)
9. [Prediction on New Data](#9-prediction-on-new-data)
10. [Graphical User Interface](#10-graphical-user-interface)
11. [Project Structure](#11-project-structure)
12. [Technologies Used](#12-technologies-used)
13. [Setup and Running](#13-setup-and-running)
14. [Conclusion](#14-conclusion)

---

## 1. Problem Statement

The problem is formulated as a binary classification task.

Given pre-departure flight information, the system predicts:

```
0 → No Delay
1 → Delay
```

Input features available before the flight:

- Airline
- Departure airport
- Destination airport
- Day of the week
- Scheduled departure time
- Flight duration

---

## 2. Dataset

**Source:** Airlines dataset  
**Size:** 539,383 flight records, 9 columns

### 2.1 Features

| Feature | Description | Type |
|---|---|---|
| `id` | Unique record identifier | Identifier |
| `Airline` | Airline code | Categorical |
| `Flight` | Flight number | Numerical |
| `AirportFrom` | Departure airport code | Categorical |
| `AirportTo` | Destination airport code | Categorical |
| `DayOfWeek` | Day of the week (1-7) | Categorical |
| `Time` | Scheduled departure time in minutes from midnight | Numerical |
| `Length` | Flight duration in minutes | Numerical |
| `Delay` | Delay status — target variable | Binary |

Sample records:

```text
id,Airline,Flight,AirportFrom,AirportTo,DayOfWeek,Time,Length,Delay
1,CO,269,SFO,IAH,3,15,205,1
2,US,1558,PHX,CLT,3,15,222,1
3,AA,2400,LAX,DFW,3,20,165,1
4,AA,2466,SFO,DFW,3,20,195,1
5,AS,108,ANC,SEA,3,30,202,0
```

### 2.2 Target Distribution

| Delay | Flight Count | Percentage |
|---|---:|---:|
| 0 (No Delay) | 299,119 | 55.46% |
| 1 (Delayed) | 240,264 | 44.54% |

Both classes have substantial representation, so the dataset is reasonably balanced.

---

## 3. Exploratory Data Analysis

### 3.1 Distribution of Scheduled Departure Times

The `Time` feature stores departure time in minutes from midnight. The histogram below shows how flights are distributed across the day.

![Distribution of Scheduled Departure Times](v1/plots/distribution_of_scheduled_departure_times.png)

### 3.2 Delay Rate by Departure Hour

Departure time was converted to hours and the delay rate was calculated per hour:

```python
airlines["Hour"] = airlines["Time"] // 60

hourly_delay_rate = (
    airlines.groupby("Hour")["Delay"]
    .mean() * 100
)
```

![Delay Rate vs Scheduled Departure Time](v1/plots/delay_rate_vs_scheduled_departure_time.png)

### 3.3 Delay Rate by Airline

```python
airline_delay_rate = (
    airlines.groupby("Airline")["Delay"]
    .mean()
    .sort_values(ascending=False) * 100
)
```

![Delay Rate by Airline](v1/plots/delay_rate_by_airline.png)

### 3.4 Delay Rate by Day of Week

```python
day_delay_rate = (
    airlines.groupby("DayOfWeek")["Delay"]
    .mean() * 100
)
```

![Delay Rate by Day of Week](v1/plots/delay_rate_by_day_of_week.png)

### 3.5 Delay Rate by Flight Duration

Flight duration was divided into bins to examine the relationship with delay rate:

```python
length_bins = pd.cut(
    airlines["Length"],
    bins=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660],
    right=False
)

length_delay_rate = (
    airlines.groupby(length_bins, observed=True)["Delay"]
    .mean() * 100
)
```

![Delay Rate by Flight Duration](v1/plots/delay_rate_by_flight_duration.png)

### 3.6 Delay Rate by Departure Airport

The ten busiest departure airports were filtered and their delay rates were compared:

```python
top_departure_airports = (
    airlines["AirportFrom"].value_counts().head(10).index
)

airport_delay_rate = (
    airlines[airlines["AirportFrom"].isin(top_departure_airports)]
    .groupby("AirportFrom")["Delay"]
    .mean()
    .sort_values(ascending=False) * 100
)
```

![Delay Rate by Departure Airport](v1/plots/delay_rate_by_departure_airport.png)

---

## 4. Feature Engineering

### 4.1 Cyclical Transformation of Departure Time

Departure time is a cyclical variable. A flight at 23:50 and a flight at 00:10 are 20 minutes apart, but treating `Time` as a plain integer creates an artificial gap of nearly 1440 units between them.

To encode the circular nature of time, sine and cosine transformations were applied:

```python
airlines["Time_sin"] = np.sin(2 * np.pi * airlines["Time"] / 1440)
airlines["Time_cos"] = np.cos(2 * np.pi * airlines["Time"] / 1440)
```

This replaces the single `Time` feature with two continuous features, `Time_sin` and `Time_cos`, that correctly represent the proximity of times near midnight.

### 4.2 Feature Selection

The following columns were dropped before training:

```python
airlines.drop(columns=["id", "Flight", "Time", "Hour"], inplace=True)
```

| Column | Reason for removal |
|---|---|
| `id` | Record identifier only, carries no predictive information |
| `Flight` | Flight number does not generalize across records |
| `Time` | Replaced by `Time_sin` and `Time_cos` |
| `Hour` | Created only for EDA, redundant with cyclical features |

After this step:

```text
X shape: (539383, 7)
y shape: (539383,)
```

---

## 5. Preprocessing

### 5.1 Train-Test Split

An 80/20 stratified split was applied:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

| Split | Shape |
|---|---:|
| `X_train` | 431,506 x 7 |
| `X_test` | 107,877 x 7 |

`stratify=y` ensures both splits preserve the original 55/45 class ratio.

### 5.2 ColumnTransformer Pipeline

Two transformations were applied depending on feature type:

| Feature Type | Features | Transformation |
|---|---|---|
| Categorical | `Airline`, `AirportFrom`, `AirportTo`, `DayOfWeek` | OneHotEncoder |
| Numerical | `Length`, `Time_sin`, `Time_cos` | StandardScaler |

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Airline", "AirportFrom", "AirportTo", "DayOfWeek"]
        ),
        (
            "numerical",
            StandardScaler(),
            ["Length", "Time_sin", "Time_cos"]
        )
    ]
)
```

The preprocessor was fitted only on training data, then applied to both splits:

```python
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)
```

After preprocessing:

```text
X_train_processed: (431506, 614)
X_test_processed:  (107877, 614)
```

One-Hot Encoding expanded the categorical columns from 4 to 611 binary columns. Together with the 3 scaled numerical features, the final feature matrix has **614 features**.

---

## 6. Model Training

Five classification models were formally trained and compared on the same preprocessed dataset.

### 6.1 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_processed, y_train)
```

### 6.2 Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_processed, y_train)
```

### 6.3 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
random_forest.fit(X_train_processed, y_train)
```

### 6.4 K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_processed, y_train)
```

### 6.5 Support Vector Machine

A linear SVM was trained using `LinearSVC`:

```python
from sklearn.svm import LinearSVC

svm_model = LinearSVC(random_state=42, max_iter=1000)
svm_model.fit(X_train_processed, y_train)
```

---

## 7. Model Evaluation

All models were evaluated on the unseen test set using five metrics.

### 7.1 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 65.04% | 63.68% | 50.05% | 56.05% | 69.70% |
| Decision Tree | 60.76% | 57.22% | 47.20% | 51.73% | 61.30% |
| Random Forest | 61.53% | 57.11% | 54.74% | 55.90% | 65.18% |
| KNN | 63.01% | 59.10% | 55.10% | 57.03% | 66.16% |
| SVM | 65.03% | 63.90% | 49.42% | 55.73% | 69.68% |

### 7.2 Accuracy Comparison

![Model Accuracy Comparison](v1/plots/model_accuracy_comparison.png)

### 7.3 Overall Performance Comparison

![Model Performance Comparison](v1/plots/model_performance_comparison.png)

### 7.4 Confusion Matrix (Logistic Regression)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_logistic)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["no delay", "delay"]
)
disp.plot()
```

![Logistic Regression Confusion Matrix](v1/plots/logistic_regression_confusion_matrix.png)

### 7.5 ROC Curve (Logistic Regression)

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_prob_logistic)
roc_auc = roc_auc_score(y_test, y_prob_logistic)
```

![Logistic Regression ROC Curve](v1/plots/logistic_regression_roc_curve.png)

ROC-AUC for Logistic Regression: **0.697**

### 7.6 Analysis

Logistic Regression and SVM achieved similar overall performance with accuracy and ROC-AUC both near 65% and 69.7% respectively. KNN achieved the highest F1-score (57.03%) among the five models, indicating a better balance between precision and recall. Decision Tree produced the lowest scores across all metrics.

The relatively modest accuracy across all models reflects that the available features (airline, airports, day, time, duration) provide a useful but incomplete picture of actual delay causes, many of which depend on real-time operational factors not present in this dataset.

---

## 8. Model Persistence

The preprocessor and all five trained models were saved using Joblib:

```python
import joblib

joblib.dump(preprocessor,   "models/preprocessor.pkl")
joblib.dump(logistic_model, "models/logistic_regression.pkl")
joblib.dump(decision_tree,  "models/decision_tree.pkl")
joblib.dump(random_forest,  "models/random_forest.pkl")
joblib.dump(knn_model,      "models/knn.pkl")
joblib.dump(svm_model,      "models/svm.pkl")
```

Loading them back:

```python
preprocessor_loaded  = joblib.load("models/preprocessor.pkl")
logistic_loaded      = joblib.load("models/logistic_regression.pkl")
decision_tree_loaded = joblib.load("models/decision_tree.pkl")
random_forest_loaded = joblib.load("models/random_forest.pkl")
knn_loaded           = joblib.load("models/knn.pkl")
svm_loaded           = joblib.load("models/svm.pkl")
```

Saving the preprocessor alongside the models is critical. Any new input must pass through the same fitted OneHotEncoder and StandardScaler before it reaches the model. Using a freshly fitted preprocessor would produce incorrect feature vectors.

---

## 9. Prediction on New Data

A new flight is represented as a DataFrame matching the training feature schema:

```python
input_data = pd.DataFrame({
    "Airline":    ["WN"],
    "AirportFrom": ["ATL"],
    "AirportTo":  ["LAX"],
    "DayOfWeek":  [1],
    "Length":     [240],
    "Time_sin":   [np.sin(2 * np.pi * 900 / 1440)],
    "Time_cos":   [np.cos(2 * np.pi * 900 / 1440)]
})
```

The saved preprocessor transforms it:

```python
input_processed = preprocessor_loaded.transform(input_data)
```

The model then predicts:

```python
prediction        = logistic_loaded.predict(input_processed)[0]
delay_probability = logistic_loaded.predict_proba(input_processed)[0][1]

print("Prediction:", "Delayed" if prediction == 1 else "Not Delayed")
print(f"Probability of delay: {delay_probability:.2%}")
```

---

## 10. Graphical User Interface

A Tkinter-based GUI was built to allow users to interact with the prediction system without opening the notebook.

![Flight Delay Prediction GUI](demo/gui.png)

The GUI accepts flight details through form inputs, builds the feature vector, applies the saved preprocessor, and displays the predicted delay status and probability.

### GUI Workflow

```
Flight Information Input
         |
         v
   Input Validation
         |
         v
  Feature Construction
   (Time -> Time_sin, Time_cos)
         |
         v
 Saved Preprocessor Transform
         |
         v
  Trained Model Prediction
         |
         v
  Display: Delayed / Not Delayed
         + Delay Probability
```

To launch:

```bash
python gui.py
```

---

## 11. Project Structure

```text
flight-delay-prediction/
|
|-- data/
|   `-- Airlines.csv
|
|-- demo/
|   `-- gui.png
|
|-- models/
|   |-- preprocessor.pkl
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- random_forest.pkl
|   |-- knn.pkl
|   `-- svm.pkl
|
|-- plots/
|   |-- distribution_of_scheduled_departure_times.png
|   |-- delay_rate_vs_scheduled_departure_time.png
|   |-- delay_rate_by_airline.png
|   |-- delay_rate_by_day_of_week.png
|   |-- delay_rate_by_flight_duration.png
|   |-- delay_rate_by_departure_airport.png
|   |-- logistic_regression_confusion_matrix.png
|   |-- logistic_regression_roc_curve.png
|   |-- model_accuracy_comparison.png
|   `-- model_performance_comparison.png
|
|-- flight_delay_prediction.ipynb
|-- gui.py
|-- cli.py
|-- progress.md
|-- .gitignore
`-- README.md
```

`Airlines.csv` and the `models/` directory are excluded from the repository via `.gitignore` due to file size.

---

## 12. Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Pandas | Latest | Data loading and manipulation |
| NumPy | Latest | Numerical computation and feature engineering |
| Matplotlib | Latest | Data visualization |
| Scikit-learn | Latest | Preprocessing, ML models, evaluation |
| Joblib | Latest | Model serialization |
| Tkinter | Built-in | Graphical user interface |
| Jupyter Notebook | Latest | Development and experimentation |

---

## 13. Setup and Running

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

Open the notebook:

```bash
jupyter notebook
```

Run `flight_delay_prediction.ipynb` from top to bottom to reproduce the full workflow.

To launch the GUI (after running the notebook to generate the model files):

```bash
python gui.py
```

---

## 14. Conclusion

This project demonstrates a complete supervised machine learning pipeline for flight delay prediction.

The pipeline begins with exploratory analysis of 539,383 flight records to understand delay patterns across airlines, departure times, days of the week, and airports. Feature engineering captures the cyclical nature of departure time through sine/cosine transformation. The preprocessing pipeline applies One-Hot Encoding and StandardScaler, expanding the input to 614 features.

Five classification algorithms are trained on the same processed dataset and evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC. Logistic Regression and SVM produced the highest accuracy and ROC-AUC, while KNN achieved the best F1-score balance. The trained models and preprocessor are serialized with Joblib, enabling reuse without retraining. A Tkinter GUI connects the saved models to an interactive prediction interface.

The project demonstrates the full progression from raw flight data to an evaluated, persistent, and user-facing machine learning application.