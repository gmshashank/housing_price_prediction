import pickle
from flask import Flask, app,jsonify,request,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

# Load the model
regression_model=pickle.load(open("regression_model.pkl","rb"))
scaler=pickle.load(open("scaling.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["POST"])
def predict_api():
    data=request.json["data"]
    print(data)
    print(data.values())
    print(np.array(list(data.values())).reshape(1,-1))
    scaled_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regression_model.predict(scaled_data)
    print(output[0])

    return jsonify(output[0])

@app.route("/predict",methods=["POST"])
def predict():
    data=[float(x) for x in request.form.values()]
    scaled_input=scaler.transform(np.array(data).reshape(1,-1))
    print(scaled_input)
    output=regression_model.predict(scaled_input)[0]
    print(output)
    return render_template("home.html",prediction_text=f"The predicted House Value is {output}")

if __name__=="__main__":
    app.run(debug=True)