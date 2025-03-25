import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import pandas as pd
from datetime import datetime
from datetime import timedelta
from helperclasses import DataFetcherKAGGLE
#import your_ml_model  # Import your ML model here

# Load airports from CSV
airport_df = pd.read_csv('airports.csv')
airports = airport_df['AIRPORT'].tolist()

#Load airlines data from Kaggle
# data_fetcher = DataFetcherKAGGLE()
# flight_data = data_fetcher.fetch_flight_dataset()
# airlines_demo = flight_data['AIRLINE'].unique().tolist()

airlines = ['American Airlines', 'Delta Air Lines', 'United Airlines', 'Southwest Airlines', 'JetBlue Airways']

def predict_flight():
    # Get input values from GUI
    airline = airline_combobox.get()
    departure_date = departure_date_cal.get_date()
    origin = origin_combobox.get()
    destination = destination_combobox.get()

    # Call your ML model's prediction function
    prediction = your_ml_model.predict(airline, departure_date, origin, destination)

    # Update result labels
    delay_prob_label.config(text=f"Delay Probability: {prediction['delay_probability']:.2f}%")

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
                               mindate=datetime.today(),
                               maxdate=datetime.today() + timedelta(days=60))  # Restrict to next 60 days
departure_date_cal.set_date(datetime.today().strftime('%Y-%m-%d'))  # Set initial date to today

origin_label = ttk.Label(root, text="Origin Airport:")
origin_combobox = ttk.Combobox(root, values=airports, state="readonly")
origin_combobox.set("Select origin airport")  # Set default text

destination_label = ttk.Label(root, text="Destination Airport:")
destination_combobox = ttk.Combobox(root, values=airports, state="readonly")
destination_combobox.set("Select destination airport")  # Set default text

# Create predict button
predict_button = ttk.Button(root, text="Predict Flight", command=predict_flight)

# Create result labels
delay_prob_label = ttk.Label(root, text="Delay Probability: ")

# Layout widgets
airline_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
airline_combobox.grid(row=0, column=1, padx=5, pady=5)

departure_date_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
departure_date_cal.grid(row=1, column=1, padx=5, pady=5)

origin_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
origin_combobox.grid(row=2, column=1, padx=5, pady=5)

destination_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
destination_combobox.grid(row=3, column=1, padx=5, pady=5)

predict_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

delay_prob_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
