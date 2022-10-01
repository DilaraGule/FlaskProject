from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from math import sqrt
from scipy.spatial.distance import cityblock
from scipy.spatial import distance
import iris_model

app = Flask(__name__)

secret_key = "Secret key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///informations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Data(db.Model):
    ID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(50))
    Height = db.Column(db.Integer)
    Weight = db.Column(db.Integer)

    def __init__(self, name, height, weight):
        self.Name = name
        self.Height = height
        self.Weight = weight

    def report(self):
        return f"{self.name}:{self.ID}"


@app.before_first_request
def create_data_table():
    db.create_all()


@app.route("/", methods=["GET"])
def index():
    all_data = Data.query.all()
    return render_template('index.html', information=all_data)


@app.route("/insert", methods=['POST'])
def insert():
    if request.method == 'POST':
        name = request.form['name']
        height = request.form['height']
        weight = request.form['weight']

        my_data = Data(name, height, weight)
        db.session.add(my_data)
        db.session.commit()

        return redirect(url_for('index'))


@app.route("/update", methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        my_data = Data.query.get(request.form.get('id'))

        my_data.Name = request.form['name']
        my_data.Height = request.form['height']
        my_data.Weight = request.form['weight']

        db.session.commit()

        return redirect(url_for('index'))


@app.route('/delete/<id>/', methods=['GET', 'POST'])
def delete(id):
    my_data = Data.query.get(id)
    db.session.delete(my_data)
    db.session.commit()

    return redirect(url_for('index'))


@app.route('/covariance')
def resultcov():
    data_all = Data.query.all()

    heights = [item.Height for item in data_all]
    weights = [item.Weight for item in data_all]

    mean_h = sum(heights)/float(len(heights))
    mean_w = sum(weights)/float(len(weights))

    sub_h = [i - mean_h for i in heights]
    sub_w = [i - mean_w for i in weights]

    numerator = sum([sub_h[i] * sub_w[i] for i in range(len(sub_h))])
    denominator = len(heights) - 1
    cov = numerator/denominator

    return render_template('covariance.html', cov=cov)


@app.route('/correlation')
def resultcorr():
    data_all = Data.query.all()

    heights = [item.Height for item in data_all]
    weights = [item.Weight for item in data_all]

    mean_h = sum(heights) / float(len(heights))
    mean_w = sum(weights) / float(len(weights))

    sub_h = [i - mean_h for i in heights]
    sub_w = [i - mean_w for i in weights]

    numerator = sum([sub_h[i] * sub_w[i] for i in range(len(sub_h))])
    denominator = len(heights) - 1
    cov = numerator / denominator

    numerator_h = sum(sub_h[i]**2 for i in range(len(sub_h)))
    numerator_w = sum(sub_w[i]**2 for i in range(len(sub_w)))

    std_h = sqrt(numerator_h / denominator)
    std_w = sqrt(numerator_w / denominator)

    corr = cov / (std_h * std_w)

    return render_template('correlation.html', corr=corr)


@app.route('/manhattan')
def maneuc():
    data_all = Data.query.all()

    heights = [item.Height for item in data_all]
    weights = [item.Weight for item in data_all]

    mann = cityblock(heights,weights)
    euc = distance.euclidean(heights,weights)

    return render_template('manhattan.html', mann=mann, euc=euc)


@app.route('/decision', methods=['GET', 'POST'])
def dec():
    if request.method == 'POST':
        sepal_length = request.form['sepallen']
        sepal_width = request.form['sepalwid']
        petal_length = request.form['petallen']
        petal_width = request.form['petalwid']

        y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
        trained_model = iris_model.training_model()
        prediction_value = trained_model.predict(y_pred) # tahmin etme

        setosa = 'The flower is classified as Setosa'
        versicolor = 'The flower is classified as Versicolor'
        virginica = 'The flower is classified as Virginica'

        if prediction_value == 0:
            return render_template('decisionTree.html', setosa=setosa)
        elif prediction_value == 1:
            return render_template('decisionTree.html', versicolor=versicolor)
        else:
            return render_template('decisionTree.html', virginica=virginica)

    return render_template('decisionTree.html')


if __name__ == "__main__":
    app.run()
