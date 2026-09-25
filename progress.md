# Project Progress

## Day 1 - Project Setup and Dataset

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
- Created the initial `src/` directory.
- Created `src/inspect_data.py` for the upcoming dataset inspection step.

### Current Project Structure

```text
flight-delay-prediction/
├── data/
│   ├── airlines.csv
│   ├── airports.csv
│   └── flights.csv
├── src/
│   └── inspect_data.py
├── .venv/
├── .gitignore
└── progress.md