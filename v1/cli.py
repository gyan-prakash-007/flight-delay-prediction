import joblib
import numpy as np
import pandas as pd



preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/logistic_regression.pkl")


print("Flight Delay Prediction System")
print("-" * 35)



airline = input("Enter airline code (e.g., WN): ").strip().upper()

airport_from = input("Enter departure airport code (e.g., ATL): ").strip().upper()

airport_to = input("Enter arrival airport code (e.g., LAX): ").strip().upper()



day_of_week = int(input("Enter day of week (1=Monday, 7=Sunday): "))

if day_of_week < 1 or day_of_week > 7:
    print("Invalid day of week.")
    exit()



length = float(input("Enter flight duration in minutes (e.g., 240): "))

if length < 0:
    print("Flight duration cannot be negative.")
    exit()



departure_time = input("Enter scheduled departure time (HH:MM): ").strip()

hours, minutes = map(int, departure_time.split(":"))

if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
    print("Invalid time.")
    exit()



time = hours * 60 + minutes



time_sin = np.sin(2 * np.pi * time / 1440)
time_cos = np.cos(2 * np.pi * time / 1440)



input_data = pd.DataFrame({
    "Airline": [airline],
    "AirportFrom": [airport_from],
    "AirportTo": [airport_to],
    "DayOfWeek": [day_of_week],
    "Length": [length],
    "Time_sin": [time_sin],
    "Time_cos": [time_cos]
})



input_processed = preprocessor.transform(input_data)



prediction = model.predict(input_processed)[0]
delay_probability = model.predict_proba(input_processed)[0][1]



print("\nPrediction Result")
print("-" * 35)

if prediction == 1:
    print("Prediction: Delayed")
else:
    print("Prediction: Not Delayed")

print(f"Probability of delay: {delay_probability:.2%}")