import os
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

preprocessor = joblib.load(
    os.path.join(BASE_DIR, "models", "preprocessor.pkl")
)

model = joblib.load(
    os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
)


AIRLINE_NAMES = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "CO": "Continental Airlines",
    "DL": "Delta Air Lines",
    "EV": "ExpressJet",
    "F9": "Frontier Airlines",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "US": "US Airways",
    "VX": "Virgin America",
    "WN": "Southwest Airlines",
    "XE": "ExpressJet",
    "YV": "Mesa Airlines"
}


def load_airports():
    dataset_path = os.path.join(
        BASE_DIR,
        "data",
        "Airlines.csv"
    )

    if os.path.exists(dataset_path):
        try:
            data = pd.read_csv(dataset_path)

            airports = sorted(
                set(data["AirportFrom"].dropna().unique())
                |
                set(data["AirportTo"].dropna().unique())
            )

            return airports

        except Exception:
            pass

    return [
        "ATL",
        "ORD",
        "DFW",
        "DEN",
        "LAX",
        "IAH",
        "PHX",
        "DTW",
        "LAS",
        "SFO",
        "MCO",
        "CLT",
        "MSP",
        "BOS",
        "SEA",
        "EWR",
        "JFK",
        "LGA"
    ]


airports = load_airports()


root = tk.Tk()
root.title("Flight Delay Prediction")
root.geometry("1000x700")
root.resizable(False, False)
root.configure(bg="#F5FAF7")


style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Green.Horizontal.TProgressbar",
    troughcolor="#E4F0E8",
    background="#2E8B57",
    bordercolor="#E4F0E8",
    lightcolor="#2E8B57",
    darkcolor="#2E8B57",
    thickness=10
)

style.configure(
    "Red.Horizontal.TProgressbar",
    troughcolor="#F8E4E4",
    background="#D64545",
    bordercolor="#F8E4E4",
    lightcolor="#D64545",
    darkcolor="#D64545",
    thickness=10
)

style.configure(
    "White.TCombobox",
    fieldbackground="#FFFFFF",
    background="#FFFFFF",
    foreground="#1F3027",
    bordercolor="#D7E3DC",
    arrowcolor="#263B31"
)


main_frame = tk.Frame(
    root,
    bg="#F5FAF7"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=70,
    pady=35
)


title = tk.Label(
    main_frame,
    text="✈  Flight Delay Prediction",
    font=("Helvetica", 28, "bold"),
    bg="#F5FAF7",
    fg="#123B2A"
)

title.pack()


subtitle = tk.Label(
    main_frame,
    text="Enter the flight details to predict the probability of a delay.",
    font=("Helvetica", 12),
    bg="#F5FAF7",
    fg="#687970"
)

subtitle.pack(
    pady=(5, 25)
)


form_frame = tk.Frame(
    main_frame,
    bg="#FFFFFF",
    highlightbackground="#DCE9E1",
    highlightthickness=1
)

form_frame.pack(
    fill="x"
)

form_frame.columnconfigure(0, weight=1)
form_frame.columnconfigure(1, weight=1)


def create_label(parent, text, row, column):
    label = tk.Label(
        parent,
        text=text,
        font=("Helvetica", 11, "bold"),
        bg="#FFFFFF",
        fg="#294536"
    )

    label.grid(
        row=row,
        column=column,
        sticky="w",
        padx=25,
        pady=(18, 7)
    )


airline_display_var = tk.StringVar(
    value="WN - Southwest Airlines"
)

airport_from_var = tk.StringVar(
    value="ATL"
)

airport_to_var = tk.StringVar(
    value="LAX"
)

day_var = tk.StringVar(
    value="Monday"
)

length_var = tk.StringVar(
    value="300"
)

time_var = tk.StringVar(
    value="12:00"
)


create_label(
    form_frame,
    "Airline",
    0,
    0
)

create_label(
    form_frame,
    "Departure Airport",
    0,
    1
)


airline_values = [
    f"{code} - {AIRLINE_NAMES[code]}"
    for code in sorted(AIRLINE_NAMES)
]


airline_combo = ttk.Combobox(
    form_frame,
    textvariable=airline_display_var,
    values=airline_values,
    state="readonly",
    font=("Helvetica", 11),
    style="White.TCombobox"
)

airline_combo.grid(
    row=1,
    column=0,
    sticky="ew",
    padx=(25, 12),
    ipady=6
)


airport_combo_values = airports


airport_from_combo = ttk.Combobox(
    form_frame,
    textvariable=airport_from_var,
    values=airport_combo_values,
    state="readonly",
    font=("Helvetica", 11),
    style="White.TCombobox"
)

airport_from_combo.grid(
    row=1,
    column=1,
    sticky="ew",
    padx=(12, 25),
    ipady=6
)


create_label(
    form_frame,
    "Arrival Airport",
    2,
    0
)

create_label(
    form_frame,
    "Day of Week",
    2,
    1
)


airport_to_combo = ttk.Combobox(
    form_frame,
    textvariable=airport_to_var,
    values=airport_combo_values,
    state="readonly",
    font=("Helvetica", 11),
    style="White.TCombobox"
)

airport_to_combo.grid(
    row=3,
    column=0,
    sticky="ew",
    padx=(25, 12),
    ipady=6
)


day_combo = ttk.Combobox(
    form_frame,
    textvariable=day_var,
    values=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ],
    state="readonly",
    font=("Helvetica", 11),
    style="White.TCombobox"
)

day_combo.grid(
    row=3,
    column=1,
    sticky="ew",
    padx=(12, 25),
    ipady=6
)


create_label(
    form_frame,
    "Flight Duration (minutes)",
    4,
    0
)

create_label(
    form_frame,
    "Scheduled Departure",
    4,
    1
)


length_entry = tk.Entry(
    form_frame,
    textvariable=length_var,
    font=("Helvetica", 11),
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground="#D7E3DC",
    highlightcolor="#2E8B57"
)

length_entry.grid(
    row=5,
    column=0,
    sticky="ew",
    padx=(25, 12),
    ipady=7
)


time_entry = tk.Entry(
    form_frame,
    textvariable=time_var,
    font=("Helvetica", 11),
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground="#D7E3DC",
    highlightcolor="#2E8B57"
)

time_entry.grid(
    row=5,
    column=1,
    sticky="ew",
    padx=(12, 25),
    ipady=7
)


button_frame = tk.Frame(
    main_frame,
    bg="#F5FAF7"
)

button_frame.pack(
    pady=22
)


def get_airline_code():
    return airline_display_var.get().split(" - ")[0]


def clear_fields():
    airline_display_var.set(
        "WN - Southwest Airlines"
    )

    airport_from_var.set("ATL")
    airport_to_var.set("LAX")
    day_var.set("Monday")
    length_var.set("300")
    time_var.set("12:00")

    result_title.config(
        text="Prediction",
        fg="#708077"
    )

    result_probability.config(
        text="--"
    )

    progress_bar["value"] = 0

    progress_bar.configure(
        style="Green.Horizontal.TProgressbar"
    )


def use_example():
    airline_display_var.set(
        "WN - Southwest Airlines"
    )

    airport_from_var.set("ATL")
    airport_to_var.set("LAX")
    day_var.set("Monday")
    length_var.set("300")
    time_var.set("12:00")


def predict_delay():

    airline = get_airline_code()
    airport_from = airport_from_var.get()
    airport_to = airport_to_var.get()
    day = day_var.get()
    length_text = length_var.get().strip()
    departure_time = time_var.get().strip()

    if airport_from == airport_to:
        messagebox.showerror(
            "Invalid Input",
            "Departure and arrival airports cannot be the same."
        )
        return

    try:
        length = float(length_text)

        if length <= 0 or length > 1000:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Flight duration must be between 1 and 1000 minutes."
        )
        return

    try:
        hours, minutes = map(
            int,
            departure_time.split(":")
        )

        if (
            hours < 0
            or hours > 23
            or minutes < 0
            or minutes > 59
        ):
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Enter departure time in HH:MM format."
        )
        return

    day_mapping = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7
    }

    day_of_week = day_mapping[day]

    time = hours * 60 + minutes

    time_sin = np.sin(
        2 * np.pi * time / 1440
    )

    time_cos = np.cos(
        2 * np.pi * time / 1440
    )

    input_data = pd.DataFrame({
        "Airline": [airline],
        "AirportFrom": [airport_from],
        "AirportTo": [airport_to],
        "DayOfWeek": [day_of_week],
        "Length": [length],
        "Time_sin": [time_sin],
        "Time_cos": [time_cos]
    })

    try:
        input_processed = preprocessor.transform(
            input_data
        )

        prediction = model.predict(
            input_processed
        )[0]

        delay_probability = model.predict_proba(
            input_processed
        )[0][1]

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            str(error)
        )
        return

    probability = delay_probability * 100

    progress_bar["value"] = probability

    result_probability.config(
        text=f"{probability:.2f}%"
    )

    if prediction == 1:

        result_title.config(
            text="DELAYED",
            fg="#D64545"
        )

        progress_bar.configure(
            style="Red.Horizontal.TProgressbar"
        )

    else:

        result_title.config(
            text="NOT DELAYED",
            fg="#2E8B57"
        )

        progress_bar.configure(
            style="Green.Horizontal.TProgressbar"
        )


predict_button = tk.Button(
    button_frame,
    text="Predict Flight Delay",
    command=predict_delay,
    font=("Helvetica", 11, "bold"),
    bg="#2E8B57",
    fg="#FFFFFF",
    activebackground="#256F46",
    activeforeground="#FFFFFF",
    relief="flat",
    cursor="hand2",
    padx=28,
    pady=11
)

predict_button.pack(
    side="left"
)


example_button = tk.Button(
    button_frame,
    text="Example",
    command=use_example,
    font=("Helvetica", 11),
    bg="#E8F3EC",
    fg="#2E6F49",
    activebackground="#DCEBE2",
    activeforeground="#2E6F49",
    relief="flat",
    cursor="hand2",
    padx=22,
    pady=11
)

example_button.pack(
    side="left",
    padx=10
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    font=("Helvetica", 11),
    bg="#FFFFFF",
    fg="#53645A",
    activebackground="#F0F3F1",
    activeforeground="#53645A",
    relief="solid",
    bd=1,
    cursor="hand2",
    padx=22,
    pady=10
)

clear_button.pack(
    side="left"
)


result_frame = tk.Frame(
    main_frame,
    bg="#FFFFFF",
    highlightbackground="#DCE9E1",
    highlightthickness=1
)

result_frame.pack(
    fill="x"
)


result_label = tk.Label(
    result_frame,
    text="PREDICTION RESULT",
    font=("Helvetica", 10, "bold"),
    bg="#FFFFFF",
    fg="#86A494"
)

result_label.pack(
    pady=(18, 2)
)


result_title = tk.Label(
    result_frame,
    text="Prediction",
    font=("Helvetica", 25, "bold"),
    bg="#FFFFFF",
    fg="#708077"
)

result_title.pack()


result_probability = tk.Label(
    result_frame,
    text="--",
    font=("Helvetica", 19, "bold"),
    bg="#FFFFFF",
    fg="#173B2A"
)

result_probability.pack(
    pady=(2, 10)
)


progress_bar = ttk.Progressbar(
    result_frame,
    orient="horizontal",
    length=650,
    mode="determinate",
    maximum=100,
    value=0,
    style="Green.Horizontal.TProgressbar"
)

progress_bar.pack(
    pady=(0, 22)
)


root.mainloop()