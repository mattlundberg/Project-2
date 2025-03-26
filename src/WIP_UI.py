import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import pandas as pd
from datetime import datetime
from modelhelper import ModelHelper  # Import the ModelHelper class 

print("Loading Airports")

# Load airports from CSV
airport_df = pd.read_csv('airports.csv')
airports = airport_df['AIRPORT'].tolist()

print("Loading Airlines")
# Define airlines for dropdown
airlines = ['Southwest Airlines Co.', 'Delta Air Lines Inc.', 'American Airlines Inc.', 'Spirit Air Lines', 'Allegiant Air']

print("Loading Model")
model_helper = ModelHelper()
model = model_helper.iniztialize_model('logistic_regression','classification')

def predict_flight():
    # Get input values from GUI
    airline = airline_combobox.get()
    departure_date = departure_date_cal.get_date()
    origin = origin_combobox.get()

    # Call the prepare_and_predict function
    prediction = prepare_and_predict(airline, origin, departure_date)

    # Update result labels
    result_label.config(text=f"Flight Status Prediction: {prediction.capitalize()}")


def prepare_and_predict(airline, origin, departure_date):
    # Create a dataframe from the inputs
    input_df = pd.DataFrame({
        'ORIGIN': [origin]
    })

    # Load the transformation mapping from CSV
    airports_df = pd.read_csv('airports.csv')

    # Create a mapping dictionary from AIRPORT_NAME to AIRPORT_CODE
    airport_mapping = dict(zip(airports_df['AIRPORT'], airports_df['Airport Code']))

    # Transform the ORIGIN to AIRPORT_CODE
    if origin in airport_mapping:
        input_df['ORIGIN'] = airport_mapping[origin]
    else:
        raise ValueError(f"Origin airport '{origin}' not found in the airports database.")

    # Make prediction using the ML model
    prediction = model_helper.predict(model=model, airline=airline, departure_date=departure_date, origin=input_df['ORIGIN'].iloc[0])    return prediction

    # Used for testing output
    #test = print(input_df)
    #return test

# Create main window
root = tk.Tk()
root.title("Flight Prediction")

# Create input fields
airline_label = ttk.Label(root, text="Airline:")
airline_combobox = ttk.Combobox(root, values=airlines, state="readonly")
airline_combobox.set("Select an airline")  # Set default text

departure_date_label = ttk.Label(root, text="Departure Date:")
departure_date_cal = DateEntry(root, width=12, background='darkblue',
                               foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd',
                               mindate=datetime.today())

departure_date_cal.set_date(datetime.today().strftime('%Y-%m-%d'))  # Set initial date to today

origin_label = ttk.Label(root, text="Origin Airport:")
origin_combobox = ttk.Combobox(root, values=airports, state="readonly")
origin_combobox.set("Select origin airport")  # Set default text

# Create predict button
predict_button = ttk.Button(root, text="Predict Flight", command=predict_flight)

# Create result labels
result_label = ttk.Label(root, text="Your Flight Is Likely To Be: ")

# Layout widget parameters
airline_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
airline_combobox.grid(row=0, column=1, padx=5, pady=5)

departure_date_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
departure_date_cal.grid(row=1, column=1, padx=5, pady=5)

origin_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
origin_combobox.grid(row=2, column=1, padx=5, pady=5)

predict_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

result_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()