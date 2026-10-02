import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
import os


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/logistic_regression.pkl")


# ============================================================
# COLORS
# ============================================================

BG_COLOR = "#F5FAF6"
CARD_COLOR = "#FFFFFF"
GREEN = "#176B45"
DARK_GREEN = "#0B3D2E"
LIGHT_GREEN = "#E8F5EA"
BORDER_GREEN = "#CFE3D4"
TEXT_COLOR = "#263238"
SECONDARY_TEXT = "#607278"
WHITE = "#FFFFFF"
BAR_BACKGROUND = "#D8DEE0"


# ============================================================
# AIRLINE NAMES
# ============================================================

airline_names = {
    "9E": "9E - Endeavor Air",
    "AA": "AA - American Airlines",
    "AS": "AS - Alaska Airlines",
    "B6": "B6 - JetBlue Airways",
    "CO": "CO - Continental Airlines",
    "DL": "DL - Delta Air Lines",
    "EV": "EV - ExpressJet",
    "F9": "F9 - Frontier Airlines",
    "HA": "HA - Hawaiian Airlines",
    "MQ": "MQ - Envoy Air",
    "OH": "OH - PSA Airlines",
    "OO": "OO - SkyWest Airlines",
    "UA": "UA - United Airlines",
    "US": "US - US Airways",
    "VX": "VX - Virgin America",
    "WN": "WN - Southwest Airlines",
    "XE": "XE - ExpressJet",
    "YV": "YV - Mesa Airlines"
}


# ============================================================
# AIRPORT NAMES
# ============================================================

# Common airports are displayed with their city.
# Any airport not listed here will simply display its airport code.

airport_names = {
    "ATL": "ATL - Atlanta, GA",
    "LAX": "LAX - Los Angeles, CA",
    "ORD": "ORD - Chicago, IL",
    "DFW": "DFW - Dallas, TX",
    "DEN": "DEN - Denver, CO",
    "JFK": "JFK - New York, NY",
    "SFO": "SFO - San Francisco, CA",
    "LAS": "LAS - Las Vegas, NV",
    "PHX": "PHX - Phoenix, AZ",
    "IAH": "IAH - Houston, TX",
    "MCO": "MCO - Orlando, FL",
    "SEA": "SEA - Seattle, WA",
    "CLT": "CLT - Charlotte, NC",
    "EWR": "EWR - Newark, NJ",
    "MSP": "MSP - Minneapolis, MN",
    "DTW": "DTW - Detroit, MI",
    "BOS": "BOS - Boston, MA",
    "PHL": "PHL - Philadelphia, PA",
    "LGA": "LGA - New York, NY",
    "FLL": "FLL - Fort Lauderdale, FL",
    "BWI": "BWI - Baltimore, MD",
    "DCA": "DCA - Washington, DC",
    "IAD": "IAD - Washington, DC",
    "TPA": "TPA - Tampa, FL",
    "SAN": "SAN - San Diego, CA",
    "PDX": "PDX - Portland, OR",
    "STL": "STL - St. Louis, MO",
    "BNA": "BNA - Nashville, TN",
    "AUS": "AUS - Austin, TX",
    "RDU": "RDU - Raleigh, NC",
    "DAL": "DAL - Dallas, TX",
    "HOU": "HOU - Houston, TX",
    "OAK": "OAK - Oakland, CA",
    "SJC": "SJC - San Jose, CA",
    "SMF": "SMF - Sacramento, CA",
    "MCI": "MCI - Kansas City, MO",
    "CLE": "CLE - Cleveland, OH",
    "CMH": "CMH - Columbus, OH",
    "IND": "IND - Indianapolis, IN",
    "PIT": "PIT - Pittsburgh, PA",
    "CVG": "CVG - Cincinnati, OH",
    "MSY": "MSY - New Orleans, LA",
    "JAX": "JAX - Jacksonville, FL",
    "RSW": "RSW - Fort Myers, FL",
    "SAT": "SAT - San Antonio, TX",
    "SLC": "SLC - Salt Lake City, UT"
}


# ============================================================
# GET AIRPORTS FROM DATASET
# ============================================================

def load_airports():

    try:
        data_path = "data/Airlines.csv"

        if os.path.exists(data_path):

            airport_data = pd.read_csv(
                data_path,
                usecols=["AirportFrom", "AirportTo"]
            )

            airports = sorted(
                set(airport_data["AirportFrom"].dropna().unique())
                |
                set(airport_data["AirportTo"].dropna().unique())
            )

            return airports

    except Exception:
        pass

    # Fallback list if dataset cannot be loaded
    return sorted(airport_names.keys())


airport_codes = load_airports()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_airline_code(display_value):

    return display_value.split(" - ")[0]


def get_airport_code(display_value):

    return display_value.split(" - ")[0]


def airport_display_name(code):

    if code in airport_names:
        return airport_names[code]

    return code


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Flight Delay Prediction")
root.geometry("1536x1000")
root.minsize(1100, 750)
root.configure(bg=BG_COLOR)


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use("clam")
except tk.TclError:
    pass


style.configure(
    "TCombobox",
    font=("Helvetica", 13),
    padding=10,
    fieldbackground=WHITE,
    background=WHITE,
    foreground=TEXT_COLOR
)

style.map(
    "TCombobox",
    fieldbackground=[("readonly", WHITE)],
    background=[("readonly", WHITE)]
)


# ============================================================
# MAIN CONTAINER
# ============================================================

main_container = tk.Frame(
    root,
    bg=BG_COLOR
)

main_container.pack(
    fill="both",
    expand=True,
    padx=32,
    pady=24
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    main_container,
    bg=BG_COLOR
)

header.pack(
    fill="x",
    pady=(5, 25)
)


# Plane icon
plane_label = tk.Label(
    header,
    text="✈",
    font=("Helvetica", 54),
    fg=DARK_GREEN,
    bg=BG_COLOR
)

plane_label.pack(
    side="left",
    padx=(350, 25)
)


title_container = tk.Frame(
    header,
    bg=BG_COLOR
)

title_container.pack(
    side="left"
)


title_label = tk.Label(
    title_container,
    text="Flight Delay Prediction",
    font=("Helvetica", 38, "bold"),
    fg=DARK_GREEN,
    bg=BG_COLOR
)

title_label.pack(
    anchor="w"
)


subtitle_label = tk.Label(
    title_container,
    text="ML-Based Flight Delay Prediction System",
    font=("Helvetica", 19),
    fg=SECONDARY_TEXT,
    bg=BG_COLOR
)

subtitle_label.pack(
    anchor="w",
    pady=(4, 0)
)


# ============================================================
# CONTENT AREA
# ============================================================

content = tk.Frame(
    main_container,
    bg=BG_COLOR
)

content.pack(
    fill="both",
    expand=True
)

content.grid_columnconfigure(0, weight=1)
content.grid_columnconfigure(1, weight=1)
content.grid_rowconfigure(0, weight=1)


# ============================================================
# LEFT CARD
# ============================================================

left_card = tk.Frame(
    content,
    bg=CARD_COLOR,
    highlightbackground=BORDER_GREEN,
    highlightthickness=1
)

left_card.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 12)
)


left_inner = tk.Frame(
    left_card,
    bg=CARD_COLOR
)

left_inner.pack(
    fill="both",
    expand=True,
    padx=32,
    pady=28
)


# ============================================================
# INPUT VARIABLES
# ============================================================

airline_var = tk.StringVar()

departure_var = tk.StringVar()

arrival_var = tk.StringVar()

day_var = tk.StringVar()

duration_var = tk.StringVar()

time_var = tk.StringVar()


# ============================================================
# INPUT FIELD FUNCTION
# ============================================================

def create_label(parent, text):

    label = tk.Label(
        parent,
        text=text,
        font=("Helvetica", 14),
        fg=TEXT_COLOR,
        bg=CARD_COLOR
    )

    label.pack(
        anchor="w",
        pady=(10, 6)
    )

    return label


def create_combobox(parent, variable, values):

    combo = ttk.Combobox(
        parent,
        textvariable=variable,
        values=values,
        state="readonly",
        font=("Helvetica", 13)
    )

    combo.pack(
        fill="x",
        ipady=7
    )

    return combo


# ============================================================
# AIRLINE
# ============================================================

create_label(
    left_inner,
    "Airline"
)

airline_values = [
    airline_names.get(code, code)
    for code in sorted(airline_names.keys())
]

airline_combo = create_combobox(
    left_inner,
    airline_var,
    airline_values
)

airline_combo.set("WN - Southwest Airlines")


# ============================================================
# DEPARTURE AIRPORT
# ============================================================

create_label(
    left_inner,
    "Departure Airport"
)

airport_values = [
    airport_display_name(code)
    for code in airport_codes
]

departure_combo = create_combobox(
    left_inner,
    departure_var,
    airport_values
)

if "ATL" in airport_codes:
    departure_combo.set("ATL - Atlanta, GA")
else:
    departure_combo.current(0)


# ============================================================
# ARRIVAL AIRPORT
# ============================================================

create_label(
    left_inner,
    "Arrival Airport"
)

arrival_combo = create_combobox(
    left_inner,
    arrival_var,
    airport_values
)

if "LAX" in airport_codes:
    arrival_combo.set("LAX - Los Angeles, CA")
else:
    arrival_combo.current(0)


# ============================================================
# DAY OF WEEK
# ============================================================

create_label(
    left_inner,
    "Day of Week"
)

day_values = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_combo = create_combobox(
    left_inner,
    day_var,
    day_values
)

day_combo.set("Monday")


# ============================================================
# FLIGHT DURATION
# ============================================================

create_label(
    left_inner,
    "Flight Duration (minutes)"
)

duration_entry = tk.Entry(
    left_inner,
    textvariable=duration_var,
    font=("Helvetica", 14),
    bg=WHITE,
    fg=TEXT_COLOR,
    relief="solid",
    bd=1
)

duration_entry.pack(
    fill="x",
    ipady=10
)

duration_var.set("300")


# ============================================================
# DEPARTURE TIME
# ============================================================

create_label(
    left_inner,
    "Scheduled Departure Time"
)

time_frame = tk.Frame(
    left_inner,
    bg=WHITE,
    highlightbackground="#C8CED0",
    highlightthickness=1
)

time_frame.pack(
    fill="x"
)


time_entry = tk.Entry(
    time_frame,
    textvariable=time_var,
    font=("Helvetica", 14),
    bg=WHITE,
    fg=TEXT_COLOR,
    relief="flat",
    bd=0
)

time_entry.pack(
    side="left",
    fill="x",
    expand=True,
    padx=10,
    pady=10
)

time_icon = tk.Label(
    time_frame,
    text="◷",
    font=("Helvetica", 24),
    fg=SECONDARY_TEXT,
    bg=WHITE
)

time_icon.pack(
    side="right",
    padx=12
)

time_var.set("12:00")


# ============================================================
# BUTTON AREA
# ============================================================

button_frame = tk.Frame(
    left_inner,
    bg=CARD_COLOR
)

button_frame.pack(
    fill="x",
    pady=(32, 0)
)

button_frame.grid_columnconfigure(0, weight=2)
button_frame.grid_columnconfigure(1, weight=1)
button_frame.grid_columnconfigure(2, weight=1)


# ============================================================
# RESULT VARIABLES
# ============================================================

prediction_var = tk.StringVar(
    value="READY"
)

probability_var = tk.StringVar(
    value="Probability of Delay: --"
)

model_status_var = tk.StringVar(
    value="Logistic Regression"
)


# ============================================================
# RIGHT CARD
# ============================================================

right_container = tk.Frame(
    content,
    bg=BG_COLOR
)

right_container.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(12, 0)
)

right_container.grid_rowconfigure(0, weight=1)
right_container.grid_rowconfigure(1, weight=1)
right_container.grid_rowconfigure(2, weight=1)
right_container.grid_columnconfigure(0, weight=1)


# ============================================================
# PREDICTION CARD
# ============================================================

prediction_card = tk.Frame(
    right_container,
    bg=CARD_COLOR,
    highlightbackground=BORDER_GREEN,
    highlightthickness=1
)

prediction_card.grid(
    row=0,
    column=0,
    sticky="nsew",
    pady=(0, 10)
)


prediction_inner = tk.Frame(
    prediction_card,
    bg=CARD_COLOR
)

prediction_inner.pack(
    fill="both",
    expand=True,
    padx=26,
    pady=22
)


prediction_title = tk.Label(
    prediction_inner,
    text="Prediction",
    font=("Helvetica", 19, "bold"),
    fg=TEXT_COLOR,
    bg=CARD_COLOR
)

prediction_title.pack(
    anchor="w"
)


# Result box

result_box = tk.Frame(
    prediction_inner,
    bg=LIGHT_GREEN,
    highlightbackground="#C4E0CA",
    highlightthickness=1
)

result_box.pack(
    fill="both",
    expand=True,
    pady=(12, 0)
)


prediction_label = tk.Label(
    result_box,
    textvariable=prediction_var,
    font=("Helvetica", 42, "bold"),
    fg=DARK_GREEN,
    bg=LIGHT_GREEN
)

prediction_label.pack(
    anchor="w",
    padx=38,
    pady=(25, 5)
)


probability_label = tk.Label(
    result_box,
    textvariable=probability_var,
    font=("Helvetica", 19),
    fg=DARK_GREEN,
    bg=LIGHT_GREEN
)

probability_label.pack(
    anchor="w",
    padx=38
)


# ============================================================
# PROGRESS BAR
# ============================================================

progress_frame = tk.Frame(
    result_box,
    bg=LIGHT_GREEN
)

progress_frame.pack(
    fill="x",
    padx=38,
    pady=(18, 25)
)

progress_frame.grid_columnconfigure(0, weight=1)


progress_canvas = tk.Canvas(
    progress_frame,
    height=30,
    bg=BAR_BACKGROUND,
    highlightthickness=0
)

progress_canvas.grid(
    row=0,
    column=0,
    sticky="ew"
)


percentage_label = tk.Label(
    progress_frame,
    text="--",
    font=("Helvetica", 16),
    fg=TEXT_COLOR,
    bg=LIGHT_GREEN
)

percentage_label.grid(
    row=0,
    column=1,
    padx=(12, 0)
)


# ============================================================
# MODEL INFORMATION CARD
# ============================================================

info_card = tk.Frame(
    right_container,
    bg=CARD_COLOR,
    highlightbackground=BORDER_GREEN,
    highlightthickness=1
)

info_card.grid(
    row=1,
    column=0,
    sticky="nsew",
    pady=10
)


info_inner = tk.Frame(
    info_card,
    bg=CARD_COLOR
)

info_inner.pack(
    fill="both",
    expand=True,
    padx=32,
    pady=22
)


info_title = tk.Label(
    info_inner,
    text="Model Information",
    font=("Helvetica", 18, "bold"),
    fg=TEXT_COLOR,
    bg=CARD_COLOR
)

info_title.pack(
    anchor="w",
    pady=(0, 12)
)


def info_row(label, value):

    row = tk.Frame(
        info_inner,
        bg=CARD_COLOR
    )

    row.pack(
        fill="x",
        pady=4
    )

    label_widget = tk.Label(
        row,
        text=label,
        font=("Helvetica", 13),
        fg=TEXT_COLOR,
        bg=CARD_COLOR,
        width=18,
        anchor="w"
    )

    label_widget.pack(
        side="left"
    )

    value_widget = tk.Label(
        row,
        text=value,
        font=("Helvetica", 13),
        fg=TEXT_COLOR,
        bg=CARD_COLOR,
        anchor="w",
        justify="left",
        wraplength=430
    )

    value_widget.pack(
        side="left",
        fill="x",
        expand=True
    )


info_row(
    "Model:",
    "Logistic Regression"
)

info_row(
    "Accuracy:",
    "65.04%"
)

info_row(
    "Dataset:",
    "U.S. Flights Dataset (2015)"
)

info_row(
    "Features Used:",
    "Airline, airports, day of week, flight duration, departure time"
)


# ============================================================
# DISCLAIMER CARD
# ============================================================

disclaimer_card = tk.Frame(
    right_container,
    bg=LIGHT_GREEN,
    highlightbackground=BORDER_GREEN,
    highlightthickness=1
)

disclaimer_card.grid(
    row=2,
    column=0,
    sticky="nsew",
    pady=(10, 0)
)


disclaimer_icon = tk.Label(
    disclaimer_card,
    text="i",
    font=("Helvetica", 18, "bold"),
    fg=WHITE,
    bg=GREEN,
    width=2,
    height=1
)

disclaimer_icon.pack(
    side="left",
    padx=(28, 15),
    pady=20
)


disclaimer_text_frame = tk.Frame(
    disclaimer_card,
    bg=LIGHT_GREEN
)

disclaimer_text_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 20),
    pady=15
)


disclaimer_title = tk.Label(
    disclaimer_text_frame,
    text="Disclaimer",
    font=("Helvetica", 16, "bold"),
    fg=DARK_GREEN,
    bg=LIGHT_GREEN
)

disclaimer_title.pack(
    anchor="w"
)


disclaimer_text = tk.Label(
    disclaimer_text_frame,
    text=(
        "Prediction is based on historical flight data "
        "and is not a guarantee of actual flight delays."
    ),
    font=("Helvetica", 12),
    fg=TEXT_COLOR,
    bg=LIGHT_GREEN,
    justify="left",
    wraplength=500
)

disclaimer_text.pack(
    anchor="w",
    pady=(3, 0)
)


# ============================================================
# BUTTON FUNCTIONS
# ============================================================

def update_progress(probability):

    progress_canvas.delete("all")

    width = progress_canvas.winfo_width()

    if width <= 1:
        width = 350

    filled_width = width * probability

    progress_canvas.create_rectangle(
        0,
        0,
        filled_width,
        30,
        fill=GREEN,
        outline=""
    )


def get_day_number(day):

    days = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7
    }

    return days[day]


def predict_delay():

    try:

        # --------------------------------------------
        # Get airline
        # --------------------------------------------

        airline_display = airline_var.get()

        if not airline_display:
            messagebox.showerror(
                "Invalid Input",
                "Please select an airline."
            )
            return

        airline = get_airline_code(
            airline_display
        )


        # --------------------------------------------
        # Get airports
        # --------------------------------------------

        departure_display = departure_var.get()
        arrival_display = arrival_var.get()

        if not departure_display or not arrival_display:

            messagebox.showerror(
                "Invalid Input",
                "Please select both airports."
            )

            return

        airport_from = get_airport_code(
            departure_display
        )

        airport_to = get_airport_code(
            arrival_display
        )


        # --------------------------------------------
        # Check same airport
        # --------------------------------------------

        if airport_from == airport_to:

            messagebox.showerror(
                "Invalid Input",
                "Departure and arrival airports cannot be the same."
            )

            return


        # --------------------------------------------
        # Day
        # --------------------------------------------

        day = day_var.get()

        if not day:

            messagebox.showerror(
                "Invalid Input",
                "Please select a day of the week."
            )

            return

        day_of_week = get_day_number(day)


        # --------------------------------------------
        # Duration
        # --------------------------------------------

        duration_text = duration_var.get().strip()

        if not duration_text:

            messagebox.showerror(
                "Invalid Input",
                "Please enter the flight duration."
            )

            return

        length = float(duration_text)

        if length <= 0:

            messagebox.showerror(
                "Invalid Input",
                "Flight duration must be greater than 0."
            )

            return


        # --------------------------------------------
        # Time
        # --------------------------------------------

        departure_time = time_var.get().strip()

        parts = departure_time.split(":")

        if len(parts) != 2:

            messagebox.showerror(
                "Invalid Time",
                "Please enter time in HH:MM format."
            )

            return

        hours = int(parts[0])
        minutes = int(parts[1])

        if (
            hours < 0
            or hours > 23
            or minutes < 0
            or minutes > 59
        ):

            messagebox.showerror(
                "Invalid Time",
                "Please enter a valid time between 00:00 and 23:59."
            )

            return


        # --------------------------------------------
        # Convert time into minutes
        # --------------------------------------------

        time = hours * 60 + minutes


        # --------------------------------------------
        # Cyclic time features
        # --------------------------------------------

        time_sin = np.sin(
            2 * np.pi * time / 1440
        )

        time_cos = np.cos(
            2 * np.pi * time / 1440
        )


        # --------------------------------------------
        # Create input dataframe
        # --------------------------------------------

        input_data = pd.DataFrame({

            "Airline": [airline],

            "AirportFrom": [airport_from],

            "AirportTo": [airport_to],

            "DayOfWeek": [day_of_week],

            "Length": [length],

            "Time_sin": [time_sin],

            "Time_cos": [time_cos]
        })


        # --------------------------------------------
        # Preprocess
        # --------------------------------------------

        input_processed = preprocessor.transform(
            input_data
        )


        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        prediction = model.predict(
            input_processed
        )[0]


        probability = model.predict_proba(
            input_processed
        )[0][1]


        # --------------------------------------------
        # Update GUI
        # --------------------------------------------

        probability_percent = probability * 100

        if prediction == 1:

            prediction_var.set(
                "DELAYED"
            )

            prediction_label.config(
                fg=DARK_GREEN
            )

        else:

            prediction_var.set(
                "NOT DELAYED"
            )

            prediction_label.config(
                fg=GREEN
            )


        probability_var.set(
            f"Probability of Delay: {probability_percent:.1f}%"
        )

        percentage_label.config(
            text=f"{probability_percent:.1f}%"
        )


        # Update progress bar after window refresh

        root.after(
            50,
            lambda: update_progress(probability)
        )


    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numerical values and time."
        )

    except Exception as error:

        messagebox.showerror(
            "Prediction Error",
            f"Something went wrong:\n\n{error}"
        )


def clear_fields():

    airline_var.set(
        "WN - Southwest Airlines"
    )

    if "ATL" in airport_codes:
        departure_var.set(
            "ATL - Atlanta, GA"
        )
    else:
        departure_combo.current(0)

    if "LAX" in airport_codes:
        arrival_var.set(
            "LAX - Los Angeles, CA"
        )
    else:
        arrival_combo.current(0)

    day_var.set(
        "Monday"
    )

    duration_var.set(
        "300"
    )

    time_var.set(
        "12:00"
    )

    prediction_var.set(
        "READY"
    )

    probability_var.set(
        "Probability of Delay: --"
    )

    percentage_label.config(
        text="--"
    )

    progress_canvas.delete(
        "all"
    )


def use_example():

    airline_var.set(
        "WN - Southwest Airlines"
    )

    if "ATL" in airport_codes:
        departure_var.set(
            "ATL - Atlanta, GA"
        )

    if "LAX" in airport_codes:
        arrival_var.set(
            "LAX - Los Angeles, CA"
        )

    day_var.set(
        "Monday"
    )

    duration_var.set(
        "300"
    )

    time_var.set(
        "12:00"
    )


# ============================================================
# BUTTONS
# ============================================================

predict_button = tk.Button(
    button_frame,
    text="Predict Flight Delay",
    command=predict_delay,
    font=("Helvetica", 14, "bold"),
    fg=WHITE,
    bg=GREEN,
    activebackground=DARK_GREEN,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=13
)

predict_button.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=(0, 7)
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    font=("Helvetica", 13),
    fg=DARK_GREEN,
    bg=LIGHT_GREEN,
    activebackground=BORDER_GREEN,
    relief="flat",
    cursor="hand2",
    padx=15,
    pady=13
)

clear_button.grid(
    row=0,
    column=1,
    sticky="ew",
    padx=7
)


example_button = tk.Button(
    button_frame,
    text="Use Example",
    command=use_example,
    font=("Helvetica", 13),
    fg=DARK_GREEN,
    bg=LIGHT_GREEN,
    activebackground=BORDER_GREEN,
    relief="flat",
    cursor="hand2",
    padx=15,
    pady=13
)

example_button.grid(
    row=0,
    column=2,
    sticky="ew",
    padx=(7, 0)
)


# ============================================================
# INITIAL PROGRESS BAR
# ============================================================

root.after(
    100,
    lambda: update_progress(0)
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()