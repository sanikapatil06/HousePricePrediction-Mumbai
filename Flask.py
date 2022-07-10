import numpy as np
import pandas as pd
import pickle as pk

from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

df = pd.read_csv("E:/VS Code/Python/House pred/train.csv")
app = Flask(__name__)
pipe = pk.load(open("E:/VS Code/Python/House pred/Trained_model.pkl", "rb"))

le = LabelEncoder()
x = le.fit_transform(df['Location'])


@app.route('/')
def index():
    Locations = sorted(df["Location"].unique())
    return render_template("index.html", Locations=Locations)


@app.route("/predict", methods=["POST"])
def predict():
    n = 0
    Location = request.form.get("Location")
    Bedrooms = request.form.get("Bedrooms")
    Area = request.form.get("Area")
    New = request.form.get("B1")
    Gymnasium = request.form.get("B2")
    Gas_Connection = request.form.get("B3")
    Car_Parking = request.form.get("B4")
    Jogging = request.form.get("B5")
    Pool = request.form.get("B6")
    for i in range(414):
        if df["Location"][i] == Location:
            n = i
            break

    if New == "1":
        New = 1
    else:
        New = 0

    if Gymnasium == '1':
        Gymnasium = 1
    else:
        Gymnasium = 0

    if Gas_Connection == '1':
        Gas_Connection = 1
    else:
        Gas_Connection = 0

    if Car_Parking == '1':
        Car_Parking = 1
    else:
        Car_Parking = 0

    if Jogging == '1':
        Jogging = 1
    else:
        Jogging = 0

    if Pool == '1':
        Pool = 1
    else:
        Pool = 0

    print(Location, Bedrooms, Area, New, Gymnasium,
          Gas_Connection, Car_Parking, Jogging, Pool)

    input = pd.DataFrame([[Area, x[n], Bedrooms, New,
                           Gymnasium, Car_Parking, Gas_Connection, Jogging,
                           Pool]],
                         columns=["Location", "No. of Bedrooms", "New/Resale",
                                  "Area",
                                  "Gymnasium", "Car Parking", "Gas Connection",
                                  "Jogging " "Track",
                                  "Pool"])

    prediction = pipe.predict(input)[0] * 1e5

    return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True)
