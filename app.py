import pickle
import pandas as pd
from flask import Flask, request, render_template
from preprocess import preprocess_data

# Load model yang sudah dilatih
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            # Ambil input dari form
            data = {
                "Distributor Name": int(request.form["distributor"]),
                "Location": int(request.form["location"]),
                "Car Name": int(request.form["car_name"]),
                "Manufacturer Name": int(request.form["manufacturer_name"]),
                "Number of Seats": int(request.form["seats"]),
                "Number of Doors": int(request.form["doors"]),
                "Energy": int(request.form["energy"]),
                "Manufactured Year": int(request.form["year"]),
                "Price-$": float(request.form["price"]),
                "Mileage-KM": float(request.form["mileage"]),
                "Engine Power-HP": float(request.form["power"]),
                "Purchased Price-$": int(request.form["purchased_price"]),
                "Sold Price-$": int(request.form["sold_price"]),
                "Margin-%": int(request.form["margin"]),
                "Sales Agent Name": int(request.form["sales_agent_name"]),
                "Sales Rating": float(request.form["rating"]),
                "Sales Commission-$": float(request.form["commission"]),
                "Car Type": request.form["car_type"],
                "Gearbox": request.form["gearbox"],
                "Color": request.form["color"],
                "Feedback": int(request.form["feedback"])
            }

            df = pd.DataFrame([data])

            # Preprocessing data
            df_clean = preprocess_data(df)

            # Prediksi menggunakan model
            prediction = model.predict(df_clean)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
