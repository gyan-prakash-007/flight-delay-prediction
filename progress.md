# Project Progress

## Day 1 (24 september 2026) - Project Setup and Dataset

### Completed

- Created the `flight-delay-prediction` project directory.

- Set up a Python virtual environment using `venv`.

- Verified Python 3.14.6 and Apple Silicon (`arm64`) environment.

- Installed and verified:
  - NumPy
  - pandas
  - matplotlib
  - scikit-learn

- Created the GitHub repository:
  - `flight-delay-prediction`

- Initialized Git locally.

- Created `.gitignore`.

- Connected the local project to GitHub.

- Created the first Git commit:
  - `Set up Python project environment`

- Downloaded the 2015 Flight Delays and Cancellations dataset.

- Extracted the dataset files:
  - `flights.csv`
  - `airlines.csv`
  - `airports.csv`

- Created the `data/` directory.

- Moved the dataset files into `data/`.

- Added `data/` to `.gitignore` because `flights.csv` is approximately 565 MB.

- Created the second Git commit:
  - `Add local dataset directory to gitignore`

- Successfully pushed the project to GitHub.

- Created the single project notebook:
  - `flight_delay_prediction.ipynb`

- Decided to keep the entire project inside the single notebook.

---

## Day 2 (27 september 2026) - Data Analysis and Preprocessing

### Completed

- Loaded `flights.csv` using pandas.

- Inspected the dataset structure and column names.

- Confirmed the dataset contains:
  - 5,819,079 rows
  - 31 columns

- Checked missing values and their percentages.

- Analyzed cancelled flights:
  - 89,884 cancelled flights
  - 5,729,195 non-cancelled flights

- Analyzed `ARRIVAL_DELAY` statistics.

- Created the `DELAYED` target variable.

- Defined:
  - `0` = Arrival delay less than 15 minutes
  - `1` = Arrival delay 15 minutes or more

- Removed flights with missing `ARRIVAL_DELAY`.

- Final target distribution:
  - Not delayed: 4,650,569
  - Delayed: 1,063,439

- Created the arrival delay distribution graph.

- Created the zoomed arrival delay distribution graph.

- Created the `plots/` directory.

- Saved the graphs:
  - `plots/arrival_delay_distribution.png`
  - `plots/arrival_delay_distribution_zoomed.png`

- Selected 12 features for prediction:
  - `YEAR`
  - `MONTH`
  - `DAY`
  - `DAY_OF_WEEK`
  - `AIRLINE`
  - `FLIGHT_NUMBER`
  - `ORIGIN_AIRPORT`
  - `DESTINATION_AIRPORT`
  - `SCHEDULED_DEPARTURE`
  - `SCHEDULED_TIME`
  - `DISTANCE`
  - `SCHEDULED_ARRIVAL`

- Removed features that could cause data leakage, including:
  - `DEPARTURE_DELAY`
  - `ARRIVAL_DELAY`
  - `DEPARTURE_TIME`
  - `ARRIVAL_TIME`
  - `TAXI_OUT`
  - `TAXI_IN`
  - `AIR_TIME`
  - `ELAPSED_TIME`
  - Delay-cause columns

- Checked missing values in the selected features.

- Found only 6 missing values in `SCHEDULED_TIME`.

- Removed the 6 rows with missing `SCHEDULED_TIME`.

- Created:
  - `X` = input features
  - `y` = target variable

- Confirmed:
  - `X`: 5,714,008 rows and 12 features
  - `y`: 5,714,008 rows

- Split the dataset into training and testing data:
  - 80% training
  - 20% testing

- Used `stratify=y` to preserve the target class distribution.

- Final split:
  - Training: 4,571,206 rows
  - Testing: 1,142,802 rows

- Identified the categorical features:
  - `AIRLINE`
  - `ORIGIN_AIRPORT`
  - `DESTINATION_AIRPORT`

- Set up `OneHotEncoder` for categorical encoding.

### Current Status

Categorical encoding has been started but not yet completed.

---

## Current Project Structure

```text
flight-delay-prediction/

│
├── data/
│   ├── flights.csv
│   ├── airlines.csv
│   └── airports.csv
│
├── flight_delay_prediction.ipynb   ← EVERYTHING
│
├── models/
│
├── plots/
│   ├── arrival_delay_distribution.png
│   └── arrival_delay_distribution_zoomed.png
│
├── gui/
│   └── prediction_gui.py
│
├── .gitignore
├── progress.md
└── README.md