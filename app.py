from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import pickle

app  = Flask(__name__)

@app.route("/")
def first_route():
    return render_template("home.html")

@app.route("/about")    
def about():
    return render_template("about.html")

@app.route("/slr")
def slr():
    return render_template("slr.html")

@app.route("/slrpred", methods=['POST'])
def slrpred():
    model = pickle.load(open("SLR.pkl","rb"))
    experience = request.form.get("experience")

    salary = model.predict([[float(experience)]])

    return render_template("slrpred.html", experience = experience, salary = int(salary[0]))

@app.route("/mlr")
def mlr():
    return render_template("mlr.html")
@app.route("/mlrpred", methods=['POST']) 
def mlrpred():
    model = pickle.load(open("MLR.pkl","rb"))
    rspend = request.form.get("rspend")
    administration = request.form.get("administration")
    mspend = request.form.get("mspend")
    states = request.form.get("states")  
    profit = model.predict(pd.DataFrame(columns=["R&D Spend","Administration","Marketing Spend","State"], data=np.array([float(rspend), float(administration), float(mspend), str(states)]).reshape(1, 4)))
    return render_template("mlrpred.html", profit = int(profit))
app.run(debug=True)    