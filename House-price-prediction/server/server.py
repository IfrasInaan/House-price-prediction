from flask import Flask, request, render_template
import pickle
import json
import numpy as np
import os

app = Flask(__name__, template_folder="../view", static_folder="../client")

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "BengalurehosePrice.pickle")
columns_path = os.path.join(BASE_DIR, "model", "columns.json")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load columns
with open(columns_path, "r") as f:
    data_columns = json.load(f)["data_columns"]


@app.route("/")
def home():
    locations = data_columns[3:]   # skip sqft, bath, bhk
    return render_template("index.html", locations=locations)


@app.route("/predict", methods=["POST"])
def predict():

    sqft = float(request.form["sqft"])
    bath = int(request.form["bath"])
    bhk = int(request.form["bhk"])
    location = request.form["location"]

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    prediction = model.predict([x])[0]

    locations = data_columns[3:]
    return render_template(
        "index.html",
        prediction_text=f"Estimated Price: {round(prediction, 2)} Lakh",
        locations=locations
    )


if __name__ == "__main__":
    app.run(debug=True)