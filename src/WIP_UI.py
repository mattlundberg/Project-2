import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import pandas as pd
from datetime import datetime
from modelhelper import ModelHelper

## Initialize ModelHelper and train the model
print("Loading Model")
model_helper = ModelHelper()
model, model_scores = model_helper.train_flight_delay_model('logistic_regression')

print(model_scores)

# Format the output of `print(model)` for display
model_info_str = f"Model Type:\n  Logistic Regression\n\nModel Scores:\n" \
                 f"  Accuracy: {model_scores['accuracy']:.4f}\n" \
                 f"  Precision: {model_scores['precision']:.4f}\n" \
                 f"  Recall: {model_scores['recall']:.4f}\n" \
                 f"  F1 Score: {model_scores['f1']:.4f}"

print("Loading Airlines")
airlines = model_helper.flight_dataset['AIRLINE'].unique().tolist()

print("Loading Airports")
airport_df = pd.read_csv('airports.csv')
airports = airport_df['AIRPORT'].tolist()

def predict_flight():
    # Get input values from GUI
    airline = airline_combobox.get()
    departure_date = departure_date_cal.get_date()
    origin = origin_combobox.get()

    # Update UI with prediction and scores
    display_sequential_messages(airline, departure_date, origin)

def display_sequential_messages(airline, departure_date, origin):
    def show_message(message, next_step=None):
        prediction_label.config(text=message)
        if next_step:
            root.after(1500, next_step)  # Wait 1.5 seconds before calling the next step

    def input_departure_date():
        show_message("Inputting Departure Date...", input_origin_airport)

    def input_origin_airport():
        show_message("Inputting Origin Airport...", display_results)

    def display_results():
        # Call the prepare_and_predict function and show the results
        prediction = prepare_and_predict(airline, origin, departure_date)
        prediction_label.config(text=f"Your flight is likely to be: \n{prediction.capitalize()}")

    # Start with the first message
    show_message("Inputting Airline...", input_departure_date)

def reset_prediction():
    prediction_label.config(text="")
    prediction_label.config(text="")  # Clear any intermediate messages as well

def prepare_and_predict(airline, origin, departure_date):
    # Transform origin to airport code
    airports_df = pd.read_csv('airports.csv')
    airport_mapping = dict(zip(airports_df['AIRPORT'], airports_df['Airport Code']))
    
    if origin not in airport_mapping:
        raise ValueError(f"Origin airport '{origin}' not found in database")
        
    origin_code = airport_mapping[origin]
    
    # Make prediction
    return model_helper.predict(
        airline=airline,
        departure_date=departure_date,
        origin=origin_code
    )

# GUI Setup
root = tk.Tk()
root.title("Flight Delay Predictor")

# Input Fields
ttk.Label(root, text="Airline:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
airline_combobox = ttk.Combobox(root, values=airlines, state="readonly")
airline_combobox.set("Select airline")
airline_combobox.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(root, text="Departure Date:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
departure_date_cal = DateEntry(root, date_pattern='yyyy-mm-dd', mindate=datetime.today())
departure_date_cal.set_date(datetime.today())
departure_date_cal.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(root, text="Origin Airport:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
origin_combobox = ttk.Combobox(root, values=airports, state="readonly")
origin_combobox.set("Select airport")
origin_combobox.grid(row=2, column=1, padx=5, pady=5)

# Prediction Button
ttk.Button(root, text="Predict Flight", command=predict_flight).grid(row=3, column=0, columnspan=2, pady=10)

# Results Display
prediction_label = tk.Label(root, text="", font= ("Arial", 10, "bold"))
prediction_label.grid(row=4, column=0, columnspan=2, pady=(10))

# Model Information Display
model_info_header_label = ttk.Label(root, text="Model Information:",font=("Arial", 10))
model_info_header_label.grid(row=5, column=0, columnspan=2, padx=5, pady=(10, 0))

model_info_label = tk.Label(root, text=model_info_str, wraplength=400, justify="left")
model_info_label.grid(row=6, column=0, columnspan=2, padx=(10,0), pady=(0, 10), sticky="w")

root.mainloop()