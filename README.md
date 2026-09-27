# Flight Delay Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.14.6-8A2BE2?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-9370DB?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-7B68EE?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-8B5CF6?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-A855F7?style=for-the-badge)

A machine learning project to predict whether a flight will be delayed by 15 minutes or more, using historical flight data from 2015.

The project is being developed using Python, Pandas, NumPy, Matplotlib, and Scikit-learn. The complete workflow is being built inside a single Jupyter Notebook: `flight_delay_prediction.ipynb`.

---

## Table of Contents

- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Data Loading](#data-loading)
- [Dataset Inspection](#dataset-inspection)
- [Missing Value Analysis](#missing-value-analysis)
- [Cancellation Analysis](#cancellation-analysis)
- [Arrival Delay Analysis](#arrival-delay-analysis)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Target Variable](#target-variable)
- [Removing Missing Arrival Delays](#removing-missing-arrival-delays)
- [Feature Selection](#feature-selection)
- [Avoiding Data Leakage](#avoiding-data-leakage)
- [Data Cleaning](#data-cleaning)
- [Creating X and y](#creating-x-and-y)
- [Train-Test Split](#train-test-split)
- [Categorical Features](#categorical-features)
- [One-Hot Encoding Setup](#one-hot-encoding-setup)
- [Current Understanding](#current-understanding)
- [Current Progress](#current-progress)
- [Development Log](#development-log)
- [Current Status](#current-status)

---

## Project Objective

The goal of this project is to predict whether a flight will experience an arrival delay of 15 minutes or more. This is treated as a binary classification problem.

```text
0 -> Not delayed
1 -> Delayed by 15 minutes or more
```

The model will use information that is available before the flight, while avoiding information that would cause data leakage.

---

## Dataset

The project uses the 2015 Flight Delays and Cancellations dataset. The dataset contains:

```text
flights.csv
airlines.csv
airports.csv
```

The main file used for the machine learning workflow is `flights.csv`.

The original `flights.csv` dataset contains:

```text
5,819,079 rows
31 columns
```

Because `flights.csv` is approximately 565 MB, the `data/` directory is excluded from Git using `.gitignore`.

---

## Technologies Used

- Python 3.14.6
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- VS Code
- Git
- GitHub

The project is being developed locally on an Apple Silicon Mac.

---

## Project Structure

```text
flight-delay-prediction/

├── data/
│   ├── flights.csv
│   ├── airlines.csv
│   └── airports.csv
│
├── flight_delay_prediction.ipynb
│
├── models/
│
├── plots/
│   ├── arrival_delay_distribution.png
│   └── arrival_delay_distribution_zoomed.png
│
├── gui/
│
├── .gitignore
├── progress.md
└── README.md
```

The entire machine learning workflow is being developed inside `flight_delay_prediction.ipynb`.

---

## Data Loading

The dataset was loaded using Pandas.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

flights = pd.read_csv("data/flights.csv")

flights.head()
```

A Pandas `DtypeWarning` appeared for:

```text
ORIGIN_AIRPORT
DESTINATION_AIRPORT
```

The warning is caused by mixed data types in these columns. The dataset loaded successfully.

---

## Dataset Inspection

The dataset shape was checked using:

```python
flights.shape
```

Result:

```text
(5819079, 31)
```

Therefore, the original dataset contains 5,819,079 flight records across 31 columns. The column names were also inspected to understand the available information.

---

## Missing Value Analysis

Missing values were checked using:

```python
flights.isnull().sum()
```

Missing value percentages were also calculated. Some columns contain a large number of missing values, for example:

```text
CANCELLATION_REASON
WEATHER_DELAY
AIR_SYSTEM_DELAY
SECURITY_DELAY
AIRLINE_DELAY
LATE_AIRCRAFT_DELAY
```

These columns contain many missing values because the information does not apply to many flights. `ARRIVAL_DELAY` also contains missing values.

---

## Cancellation Analysis

The number of cancelled and non-cancelled flights was checked using:

```python
flights["CANCELLED"].value_counts()
```

Result:

```text
0    5729195
1      89884
```

Therefore, there are 5,729,195 non-cancelled flights and 89,884 cancelled flights.

Since cancelled flights do not have a meaningful actual arrival delay, flights without a known arrival delay were removed from the classification dataset.

---

## Arrival Delay Analysis

The arrival delay was analyzed using:

```python
arrival_delay = flights[
    (flights["CANCELLED"] == 0) &
    (flights["ARRIVAL_DELAY"].notna())
]["ARRIVAL_DELAY"]
```

The statistical summary showed:

```text
Mean     approx 4.4 minutes
Median   = -5 minutes
Minimum  = -87 minutes
Maximum  = 1971 minutes
```

A negative delay means that the flight arrived earlier than its scheduled arrival time. For example, a delay of -5 minutes means the flight arrived 5 minutes early, and a delay of +20 minutes means it arrived 20 minutes late.

---

## Exploratory Data Analysis

Two arrival delay graphs have been created so far.

### 1. Arrival Delay Distribution

The first graph shows the complete distribution of arrival delays.

```python
plt.figure(figsize=(10, 6))

plt.hist(arrival_delay, bins=100)

plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Number of Flights")
plt.title("Distribution of Flight Arrival Delays")

plt.savefig(
    "plots/arrival_delay_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

**Observation:** The distribution is strongly right skewed. Most flights are concentrated around relatively small delays, while a smaller number of flights experience very large delays.

### 2. Zoomed Arrival Delay Distribution

A second graph was created to focus on the more common delay range.

```python
plt.figure(figsize=(10, 6))

plt.hist(arrival_delay, bins=100)

plt.xlim(-50, 200)

plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Number of Flights")
plt.title("Distribution of Flight Arrival Delays (-50 to 200 Minutes)")

plt.savefig(
    "plots/arrival_delay_distribution_zoomed.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

**Observation:** The zoomed graph makes the main concentration of flight delays easier to observe. Most flights are relatively close to their scheduled arrival time, while a smaller number experience much larger delays.

---

## Target Variable

A new column called `DELAYED` was created. The target is based on a 15 minute delay threshold.

```python
flights["DELAYED"] = (
    flights["ARRIVAL_DELAY"] >= 15
).astype(int)
```

This creates:

```text
Arrival delay < 15 minutes  -> DELAYED = 0
Arrival delay >= 15 minutes -> DELAYED = 1
```

`astype(int)` converts `False` to `0` and `True` to `1`.

---

## Removing Missing Arrival Delays

Flights with missing `ARRIVAL_DELAY` cannot be reliably classified, so they were removed:

```python
flights = flights[
    flights["ARRIVAL_DELAY"].notna()
]
```

The target was then recreated. Final target distribution:

```text
DELAYED

0    4650569
1    1063439
```

Therefore, there are 4,650,569 not-delayed flights and 1,063,439 delayed flights, for a total of 5,714,008 usable flights.

The delayed class represents a smaller portion of the dataset, so model evaluation will consider metrics other than accuracy as well.

---

## Feature Selection

The following 12 features were selected for the machine learning model:

```python
features = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL"
]
```

These features describe the planned flight information.

---

## Avoiding Data Leakage

Features that contain information about what happened during or after the flight were excluded. Examples include:

```text
DEPARTURE_TIME
DEPARTURE_DELAY
TAXI_OUT
WHEELS_OFF
ELAPSED_TIME
AIR_TIME
WHEELS_ON
TAXI_IN
ARRIVAL_TIME
ARRIVAL_DELAY
DIVERTED
CANCELLED
CANCELLATION_REASON
AIR_SYSTEM_DELAY
SECURITY_DELAY
AIRLINE_DELAY
LATE_AIRCRAFT_DELAY
WEATHER_DELAY
```

These features were not selected because they would provide information that would not be available when making a pre-flight prediction. `TAIL_NUMBER` was also excluded from the initial feature set.

---

## Data Cleaning

The selected features were checked for missing values. Only 6 rows had a missing value in `SCHEDULED_TIME`. These rows were removed using:

```python
flights = flights.dropna(
    subset=["SCHEDULED_TIME"]
)
```

After this step, the selected features had no missing values.

---

## Creating X and y

The input features and target variable were separated.

```python
X = flights[features]

y = flights["DELAYED"]
```

Where `X` is the input features and `y` is the target variable.

The resulting shapes were:

```text
X = (5714008, 12)
y = (5714008,)
```

This means the dataset has 5,714,008 flight records and 12 input features.

---

## Train-Test Split

The dataset was divided into training and testing data, using an 80 percent training and 20 percent testing split.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The resulting dataset sizes were:

```text
Training features -> 4,571,206 x 12
Testing features  -> 1,142,802 x 12
Training target   -> 4,571,206
Testing target    -> 1,142,802
```

**`random_state=42`** makes the split repeatable. Running the same code again produces the same split.

**`stratify=y`** helps maintain the same proportion of delayed and non-delayed flights in both the training and testing datasets.

---

## Categorical Features

Three selected features contain categorical values:

```text
AIRLINE
ORIGIN_AIRPORT
DESTINATION_AIRPORT
```

For example, `AIRLINE` contains values like `AA`, `DL`, `UA`, `AS`.

Machine learning models cannot directly use these text categories in their raw form, so One-Hot Encoding will be used.

---

## One-Hot Encoding Setup

The encoder was initialized using:

```python
from sklearn.preprocessing import OneHotEncoder

categorical_features = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT"
]

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)
```

The actual transformation of the training and testing data has not been completed yet. This is the next step of the project.

---

## Current Understanding

**Dataset:** The original dataset contains more than 5.8 million flight records.

**Arrival Delays:** Arrival delays are strongly right skewed. Most flights have relatively small delays, while a smaller number experience very large delays.

**Target:** The project uses a 15 minute threshold, where 0 means a delay under 15 minutes and 1 means a delay of 15 minutes or more.

**Target Distribution:**

```text
Not delayed -> 4,650,569
Delayed     -> 1,063,439
```

**Data Leakage:** Information that would only be available during or after the flight is excluded from the input features.

**Features:** 12 pre-flight features have been selected for the initial machine learning workflow.

---

## Current Progress

### Completed

- Project setup
- Python virtual environment
- GitHub repository
- Dataset download
- Dataset organization
- Jupyter Notebook setup
- Dataset loading
- Dataset inspection
- Missing value analysis
- Cancellation analysis
- Arrival delay analysis
- Initial exploratory data analysis
- Two arrival delay graphs
- Target variable creation
- Missing target removal
- Feature selection
- Data leakage prevention
- Missing value cleaning
- Creation of `X` and `y`
- Train-test split
- Identification of categorical features
- One-Hot Encoder setup

### Currently Working On

```text
One-Hot Encoding
```

### Next Steps

```text
Complete One-Hot Encoding
        |
Prepare final training data
        |
Additional EDA
        |
Train 5 ML models
        |
Evaluate models
        |
Select top 3 models
        |
Average predicted probabilities
        |
Final prediction
        |
Save models
        |
Build GUI
```

---

## Development Log

### Day 1 - September 26, 2026

Completed:

- Project setup
- Python environment
- GitHub repository
- Dataset download
- Dataset organization
- `.gitignore`
- Jupyter Notebook setup

### Day 2 - September 27, 2026

Completed:

- Dataset inspection
- Missing value analysis
- Cancellation analysis
- Arrival delay analysis
- Initial EDA
- Arrival delay graphs
- Target variable creation
- Feature selection
- Data leakage prevention
- Data cleaning
- `X` and `y` creation
- Train-test split
- Categorical feature identification
- One-Hot Encoder setup

---

## Current Status

The project has completed the initial data analysis and preparation stage. The next task is to complete categorical encoding and prepare the dataset for machine learning model training.