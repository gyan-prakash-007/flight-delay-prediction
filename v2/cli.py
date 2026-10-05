import os
import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# Load models
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
hgb_model = joblib.load(os.path.join(MODEL_DIR, "hist_gradient_boosting.pkl"))
knn_model = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
ensemble_threshold = joblib.load(
    os.path.join(MODEL_DIR, "ensemble_threshold.pkl")
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_integer(prompt, minimum, maximum):
    while True:
        try:
            value = int(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(
                f"Please enter a value between "
                f"{minimum} and {maximum}."
            )

        except ValueError:
            print("Please enter a valid number.")


def get_time(prompt):
    while True:
        value = input(prompt).strip()

        try:
            hour, minute = map(int, value.split(":"))

            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour, minute

            print("Please enter a valid time between 00:00 and 23:59.")

        except ValueError:
            print("Please use HH:MM format, for example 14:30.")


def get_positive_number(prompt):
    while True:
        try:
            value = float(input(prompt))

            if value > 0:
                return value

            print("Please enter a value greater than 0.")

        except ValueError:
            print("Please enter a valid number.")


# --------------------------------------------------
# Main program
# --------------------------------------------------

print()
print("=" * 46)
print("       FLIGHT DELAY PREDICTION SYSTEM")
print("=" * 46)

print("\nModels loaded successfully.")

# --------------------------------------------------
# Flight details
# --------------------------------------------------

print("\nEnter Flight Details")
print("--------------------")

airline = input("Airline code (e.g., UA): ").strip().upper()
origin = input("Departure airport code (e.g., JFK): ").strip().upper()
destination = input("Destination airport code (e.g., LAX): ").strip().upper()


# --------------------------------------------------
# Schedule details
# --------------------------------------------------

print("\nEnter Schedule Details")
print("----------------------")

month = get_integer("Month (1-12): ", 1, 12)
day = get_integer("Day of month (1-31): ", 1, 31)

dep_hour, dep_minute = get_time(
    "Scheduled departure time (HH:MM): "
)

arr_hour, arr_minute = get_time(
    "Scheduled arrival time (HH:MM): "
)

distance = get_positive_number(
    "Flight distance (miles): "
)


# --------------------------------------------------
# Convert time
# --------------------------------------------------

sched_dep_time = dep_hour * 100 + dep_minute
sched_arr_time = arr_hour * 100 + arr_minute

dep_minutes = dep_hour * 60 + dep_minute
arr_minutes = arr_hour * 60 + arr_minute


# --------------------------------------------------
# Weather values
# --------------------------------------------------
# Representative values used internally.
# These are not entered by the user.
# --------------------------------------------------

temp = 15.0
dewp = 8.0
humid = 60.0
wind_dir = 180.0
wind_speed = 10.0
precip = 0.0
pressure = 1015.0
visib = 10.0


# --------------------------------------------------
# Create input DataFrame
# --------------------------------------------------

X = pd.DataFrame([{
    "carrier": airline,
    "origin": origin,
    "dest": destination,

    "month": month,
    "day": day,

    "sched_dep_time": sched_dep_time,
    "sched_arr_time": sched_arr_time,

    "distance": distance,

    "hour": dep_hour,
    "minute": dep_minute,

    "temp": temp,
    "dewp": dewp,
    "humid": humid,
    "wind_dir": wind_dir,
    "wind_speed": wind_speed,
    "precip": precip,
    "pressure": pressure,
    "visib": visib
}])


# --------------------------------------------------
# Model predictions
# --------------------------------------------------

rf_probability = rf_model.predict_proba(X)[0, 1]

hgb_probability = hgb_model.predict_proba(
    X[[
        "carrier",
        "origin",
        "dest",
        "month",
        "day",
        "sched_dep_time",
        "sched_arr_time",
        "distance",
        "hour",
        "minute",
        "temp",
        "dewp",
        "humid",
        "wind_dir",
        "wind_speed",
        "precip",
        "pressure",
        "visib"
    ]].assign(
        carrier=lambda df: df["carrier"].astype("category"),
        origin=lambda df: df["origin"].astype("category"),
        dest=lambda df: df["dest"].astype("category")
    )
)[0, 1]

knn_probability = knn_model.predict_proba(X)[0, 1]


# --------------------------------------------------
# Ensemble prediction
# --------------------------------------------------

ensemble_probability = (
    rf_probability
    + hgb_probability
    + knn_probability
) / 3

prediction = (
    ensemble_probability >= ensemble_threshold
)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print()
print("-" * 46)
print("Prediction Result")
print("-" * 46)

if prediction:
    print("Prediction: FLIGHT DELAYED")
else:
    print("Prediction: FLIGHT ON TIME")

print(
    f"Delay probability: "
    f"{ensemble_probability * 100:.2f}%"
)

print()
print("Individual model probabilities:")
print(
    f"Random Forest:          "
    f"{rf_probability * 100:.2f}%"
)
print(
    f"HistGradientBoosting:   "
    f"{hgb_probability * 100:.2f}%"
)
print(
    f"KNN:                    "
    f"{knn_probability * 100:.2f}%"
)

print("-" * 46)