import os
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


rf_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

hgb_model = joblib.load(
    os.path.join(MODEL_DIR, "hist_gradient_boosting.pkl")
)

knn_model = joblib.load(
    os.path.join(MODEL_DIR, "knn.pkl")
)

ensemble_threshold = joblib.load(
    os.path.join(MODEL_DIR, "ensemble_threshold.pkl")
)


AIRLINE_NAMES = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "EV": "ExpressJet",
    "F9": "Frontier Airlines",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "UA": "United Airlines",
    "US": "US Airways",
    "VX": "Virgin America",
    "WN": "Southwest Airlines",
    "YV": "Mesa Airlines",
    "FL": "AirTran Airways",
    "OO": "SkyWest Airlines"
}


ORIGIN_AIRPORTS = [
    "EWR",
    "JFK",
    "LGA"
]


DESTINATION_AIRPORTS = [
    "ABQ",
    "ALB",
    "ATL",
    "AUS",
    "AVL",
    "BDL",
    "BGR",
    "BHM",
    "BNA",
    "BOS",
    "BQN",
    "BTV",
    "BUF",
    "BUR",
    "BWI",
    "CAK",
    "CHO",
    "CHS",
    "CLE",
    "CLT",
    "CMH",
    "CRW",
    "CVG",
    "DAY",
    "DCA",
    "DEN",
    "DFW",
    "DSM",
    "DTW",
    "EGE",
    "FLL",
    "GRR",
    "GSO",
    "GSP",
    "HDN",
    "HNL",
    "IAD",
    "IAH",
    "ILM",
    "IND",
    "JAC",
    "JAX",
    "LAS",
    "LAX",
    "LEX",
    "MCI",
    "MCO",
    "MEM",
    "MHT",
    "MIA",
    "MKE",
    "MSN",
    "MSP",
    "MSY",
    "MTJ",
    "MVY",
    "MYR",
    "OAK",
    "OKC",
    "OMA",
    "ORD",
    "ORF",
    "PBI",
    "PDX",
    "PHL",
    "PHX",
    "PIT",
    "PNS",
    "PWM",
    "RDU",
    "RIC",
    "ROC",
    "RSW",
    "SAN",
    "SAT",
    "SAV",
    "SDF",
    "SEA",
    "SFO",
    "SJU",
    "SLC",
    "SMF",
    "SNA",
    "SRQ",
    "STL",
    "STT",
    "SYR",
    "TPA",
    "TUL",
    "TVC",
    "TYS",
    "XNA"
]


root = tk.Tk()
root.title("Flight Delay Prediction")
root.geometry("1000x760")
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
    pady=30
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
    text="ML-Based Flight Delay Prediction System",
    font=("Helvetica", 12),
    bg="#F5FAF7",
    fg="#687970"
)

subtitle.pack(
    pady=(5, 22)
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
        pady=(14, 6)
    )


airline_var = tk.StringVar(
    value="DL - Delta Air Lines"
)

airport_from_var = tk.StringVar(
    value="JFK"
)

airport_to_var = tk.StringVar(
    value="LGA"
)

month_var = tk.StringVar(
    value="6"
)

day_var = tk.StringVar(
    value="15"
)

departure_var = tk.StringVar(
    value="12:00"
)

arrival_var = tk.StringVar(
    value="15:30"
)

distance_var = tk.StringVar(
    value="300"
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
    textvariable=airline_var,
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


airport_from_combo = ttk.Combobox(
    form_frame,
    textvariable=airport_from_var,
    values=ORIGIN_AIRPORTS,
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
    "Month",
    2,
    1
)


airport_to_combo = ttk.Combobox(
    form_frame,
    textvariable=airport_to_var,
    values=DESTINATION_AIRPORTS,
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


month_combo = ttk.Combobox(
    form_frame,
    textvariable=month_var,
    values=[str(i) for i in range(1, 13)],
    state="readonly",
    font=("Helvetica", 11),
    style="White.TCombobox"
)

month_combo.grid(
    row=3,
    column=1,
    sticky="ew",
    padx=(12, 25),
    ipady=6
)


create_label(
    form_frame,
    "Day",
    4,
    0
)

create_label(
    form_frame,
    "Distance (miles)",
    4,
    1
)


day_entry = tk.Entry(
    form_frame,
    textvariable=day_var,
    font=("Helvetica", 11),
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground="#D7E3DC",
    highlightcolor="#2E8B57"
)

day_entry.grid(
    row=5,
    column=0,
    sticky="ew",
    padx=(25, 12),
    ipady=7
)


distance_entry = tk.Entry(
    form_frame,
    textvariable=distance_var,
    font=("Helvetica", 11),
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground="#D7E3DC",
    highlightcolor="#2E8B57"
)

distance_entry.grid(
    row=5,
    column=1,
    sticky="ew",
    padx=(12, 25),
    ipady=7
)


create_label(
    form_frame,
    "Scheduled Departure",
    6,
    0
)

create_label(
    form_frame,
    "Scheduled Arrival",
    6,
    1
)


departure_entry = tk.Entry(
    form_frame,
    textvariable=departure_var,
    font=("Helvetica", 11),
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground="#D7E3DC",
    highlightcolor="#2E8B57"
)

departure_entry.grid(
    row=7,
    column=0,
    sticky="ew",
    padx=(25, 12),
    ipady=7
)


arrival_entry = tk.Entry(
    form_frame,
    textvariable=arrival_var,
    font=("Helvetica", 11),
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground="#D7E3DC",
    highlightcolor="#2E8B57"
)

arrival_entry.grid(
    row=7,
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
    pady=18
)


def get_airline_code():
    return airline_var.get().split(" - ")[0]


def use_example():
    airline_var.set("DL - Delta Air Lines")
    airport_from_var.set("JFK")
    airport_to_var.set("LGA")
    month_var.set("6")
    day_var.set("15")
    departure_var.set("12:00")
    arrival_var.set("15:30")
    distance_var.set("300")


def clear_fields():
    airline_var.set("DL - Delta Air Lines")
    airport_from_var.set("JFK")
    airport_to_var.set("LGA")
    month_var.set("6")
    day_var.set("15")
    departure_var.set("12:00")
    arrival_var.set("15:30")
    distance_var.set("300")

    result_title.config(
        text="Prediction",
        fg="#708077"
    )

    result_probability.config(
        text="--"
    )

    threshold_text.config(
        text=f"Decision threshold: {ensemble_threshold:.0%}"
    )

    rf_label.config(
        text="Random Forest: --"
    )

    hgb_label.config(
        text="HistGradientBoosting: --"
    )

    knn_label.config(
        text="KNN: --"
    )

    progress_bar["value"] = 0

    progress_bar.configure(
        style="Green.Horizontal.TProgressbar"
    )


def predict_delay():

    airline = get_airline_code()
    airport_from = airport_from_var.get()
    airport_to = airport_to_var.get()

    month_text = month_var.get().strip()
    day_text = day_var.get().strip()

    departure_time = departure_var.get().strip()
    arrival_time = arrival_var.get().strip()

    distance_text = distance_var.get().strip()

    if airport_from == airport_to:
        messagebox.showerror(
            "Invalid Input",
            "Departure and arrival airports cannot be the same."
        )
        return

    try:
        month = int(month_text)

        if month < 1 or month > 12:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Month must be between 1 and 12."
        )
        return

    try:
        day = int(day_text)

        if day < 1 or day > 31:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Day must be between 1 and 31."
        )
        return

    try:
        distance = float(distance_text)

        if distance <= 0:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Distance must be greater than 0."
        )
        return

    try:
        departure_hour, departure_minute = map(
            int,
            departure_time.split(":")
        )

        if (
            departure_hour < 0
            or departure_hour > 23
            or departure_minute < 0
            or departure_minute > 59
        ):
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Enter departure time in HH:MM format."
        )
        return

    try:
        arrival_hour, arrival_minute = map(
            int,
            arrival_time.split(":")
        )

        if (
            arrival_hour < 0
            or arrival_hour > 23
            or arrival_minute < 0
            or arrival_minute > 59
        ):
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Enter arrival time in HH:MM format."
        )
        return

    sched_dep_time = (
        departure_hour * 100
        + departure_minute
    )

    sched_arr_time = (
        arrival_hour * 100
        + arrival_minute
    )

    hour = departure_hour
    minute = departure_minute

    temp = 15.0
    dewp = 8.0
    humid = 60.0
    wind_dir = 180.0
    wind_speed = 10.0
    precip = 0.0
    pressure = 1015.0
    visib = 10.0

    input_data = pd.DataFrame({
        "month": [month],
        "day": [day],
        "sched_dep_time": [sched_dep_time],
        "sched_arr_time": [sched_arr_time],
        "carrier": [airline],
        "flight": [0],
        "tailnum": ["UNKNOWN"],
        "origin": [airport_from],
        "dest": [airport_to],
        "distance": [distance],
        "hour": [hour],
        "minute": [minute],
        "temp": [temp],
        "dewp": [dewp],
        "humid": [humid],
        "wind_dir": [wind_dir],
        "wind_speed": [wind_speed],
        "precip": [precip],
        "pressure": [pressure],
        "visib": [visib]
    })

    try:
        rf_probability = (
            rf_model.predict_proba(input_data)[0][1]
        )

        hgb_probability = (
            hgb_model.predict_proba(input_data)[0][1]
        )

        knn_probability = (
            knn_model.predict_proba(input_data)[0][1]
        )

        ensemble_probability = (
            rf_probability
            + hgb_probability
            + knn_probability
        ) / 3

        prediction = (
            ensemble_probability
            >= ensemble_threshold
        )

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            str(error)
        )
        return

    probability = ensemble_probability * 100

    progress_bar["value"] = probability

    result_probability.config(
        text=f"{probability:.2f}%"
    )

    threshold_text.config(
        text=f"Decision threshold: {ensemble_threshold:.0%}"
    )

    rf_label.config(
        text=f"Random Forest: {rf_probability:.2%}"
    )

    hgb_label.config(
        text=f"HistGradientBoosting: {hgb_probability:.2%}"
    )

    knn_label.config(
        text=f"KNN: {knn_probability:.2%}"
    )

    if prediction:
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
    pady=(14, 2)
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
    pady=(2, 3)
)


threshold_text = tk.Label(
    result_frame,
    text=f"Decision threshold: {ensemble_threshold:.0%}",
    font=("Helvetica", 10),
    bg="#FFFFFF",
    fg="#708077"
)

threshold_text.pack(
    pady=(0, 10)
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
    pady=(0, 15)
)


models_frame = tk.Frame(
    result_frame,
    bg="#FFFFFF"
)

models_frame.pack(
    pady=(0, 16)
)


rf_label = tk.Label(
    models_frame,
    text="Random Forest: --",
    font=("Helvetica", 10),
    bg="#FFFFFF",
    fg="#53645A"
)

rf_label.grid(
    row=0,
    column=0,
    padx=20
)


hgb_label = tk.Label(
    models_frame,
    text="HistGradientBoosting: --",
    font=("Helvetica", 10),
    bg="#FFFFFF",
    fg="#53645A"
)

hgb_label.grid(
    row=0,
    column=1,
    padx=20
)


knn_label = tk.Label(
    models_frame,
    text="KNN: --",
    font=("Helvetica", 10),
    bg="#FFFFFF",
    fg="#53645A"
)

knn_label.grid(
    row=0,
    column=2,
    padx=20
)


root.mainloop()