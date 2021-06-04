from flask import Flask, jsonify, request
from classifier import get_pridiction

app = Flask(__name__)

@app_route("/predict_digit", methods = ["Post"])
def predict_data():
    image = request.files.get("digit")
    prediction = get_pridiction(image)
    return jsonify({
        "prediction" : prediction
}),200

if __name__ == "__main__":
    app.run(debug =True)