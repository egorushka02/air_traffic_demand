from flask import Flask, render_template, url_for, request, redirect, send_file


from learning import train_models
from predict import get_prediction

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
    return render_template("index.html")


@app.route('/forecast')
def forecast():
    return render_template("forecast.html")


@app.route('/forecast/download', methods=["POST", "GET"])
def download():
    return send_file("documents/doc.pdf")


@app.route('/forecast/download_xlsx', methods=["POST", "GET"])
def download_xlsx():
    return send_file("statistics/prediction.xlsx")


@app.route('/learning', methods=["POST", "GET"])
def learning():
    if request.method == "POST":
        min_num_estimators = request.form['min_num_estimators']
        max_num_estimators = request.form['max_num_estimators']
        min_depth = request.form['min_depth']
        max_depth = request.form['max_depth']
        train_models(min_num_estimators, max_num_estimators, min_depth, max_depth)
        get_prediction()
        return redirect('/forecast')
    else:
        return render_template("learning.html")


@app.route('/models', methods=["POST", "GET"])
def models():
    try:
        return send_file("statistics/stat.xlsx")
    except:
        return render_template("models.html")


if __name__ == "__main__":
    app.run(debug=True)