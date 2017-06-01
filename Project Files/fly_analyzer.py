import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import flask
from flask import Flask, send_file

rf = RandomForestClassifier()
X = pickle.load(open('fly_df_X.p', 'rb'))
y = pickle.load(open('fly_df_y.p', 'rb'))
# X = df[['vote1', 'vote2', 'vote3']]
# y = df['party']
rf.fit(X,y)

app = Flask(__name__)

#this is the main page of the app
@app.route('/')
def open_home_page():
    with open("index3.html", "r") as viz:
        return viz.read()

#this opens the survey page
@app.route('/survey', methods=['POST'])
def open_survey():
    with open("index6.html", "r") as viz:
        return viz.read()

    
#this opens the final results page
@app.route('/results', methods = ['POST'])
# def open_gauge():
#     with open("index7.html", "r") as viz:
#         return viz.read()

def analyze_data():
    # data = flask.request.json
    # form = flask.request.form.getlist("q19")
    form = flask.request.form
    # print(data)
    print(form)
    # value = form.getlist("q19")
    # female = "The value of female is %s" % value[0]
    form_dict = form.to_dict()
    df = pd.DataFrame(form_dict, index=[0])
    print(df)
    x = df
    # x = np.matrix(data["results"])
    pred = rf.predict(x)[0]
    print(pred)
    print(type(pred))
    if pred == 1.0:
        filename = 'ahole.jpg'
    else:
        filename = 'ok.jpg'
    # result = {"pred": val}
    return send_file(filename, mimetype='image/jpg')

if __name__ == '__main__':
    app.run()
