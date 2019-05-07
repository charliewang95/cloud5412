from flask import Flask
from flask import jsonify
import Classify

app = Flask(__name__)

@app.route("/")
def hello():
    # arr = numpy.array(3)
    # s = Classify.Predict_from_ID(705)
    return 'Team Bubble: Charlie Wang, Chia-wei Cheng, Haonan Zhong'

@app.route("/predict/<cow>")
def predict(cow):
    return jsonify(
        willCalve = Classify.Predict_from_ID(cow)
    )