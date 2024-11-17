from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = joblib.load('model/house_price_model.pkl')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user inputs
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])

        # Prepare input features and predict
        input_features = np.array([[area, bedrooms, bathrooms]])
        predicted_price = model.predict(input_features)[0]

        # Render result
        return render_template("result.html", price=round(predicted_price, 2))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
