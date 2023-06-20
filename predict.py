import pandas as pd

from fpdf import FPDF

import pickle

import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def get_prediction():
    pkl_filename = 'models/best_model.pkl'

    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)

    df = pd.read_excel("data/actual.xlsx")
    prediction = model.predict(df)
    len_x = df.shape[0]
    x = [(i+1) for i in range(len_x)]

    plt.plot(x, prediction)
    plt.xlabel("Единицы времени")
    plt.ylabel("Пассажирокилометры (млн. ПКМ)")
    plt.title("Прогнозируемые пассажирокилометры")
    plt.grid(True)
    plt.savefig('figures/figure.png')

    fpdf = FPDF()
    fpdf.add_page()
    fpdf.set_font("Arial", size=16)
    fpdf.image("images/logo_2.png", 80, 5, w=40)
    fpdf.text(15, 40, txt = "Air traffic demand forecast, mln RPK")
    fpdf.text(15, 45, txt = "Predictive model: " + str(model))
    fpdf.text(15, 50, txt="Forecast period: " + str(df.shape[0]) + " months")
    fpdf.text(15, 70, txt="period  |  value")

    for i in range(len(x)):
        fpdf.text(15, 75+7*i, txt=str(i+1) + "       | " + str(round(prediction[i], 3)))
        fpdf.text(15, 75+7*i+2, txt="---------------------")
    fpdf.image("figures/figure.png", 75, 75, w=110)
    fpdf.output("documents/doc.pdf")
    pred = pd.DataFrame(prediction)
    pred.to_excel("statistics/prediction.xlsx")
