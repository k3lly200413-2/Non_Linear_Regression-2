import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from urllib.request import urlretrieve
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

def print_eval(X, y, model):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    re = mean_absolute_percentage_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"   Mean squared error: {mse:.5}")
    print(f"       Relative error: {re:.5%}")
    print(f"R-squared coefficient: {r2:.5}")


def plot_model_on_data(X, y, model=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(X, y)
    if model is not None:
        xlim, ylim = plt.xlim(), plt.ylim()
        line_x = np.linspace(xlim[0], xlim[1], 100)
        line_x_df = pd.DataFrame(line_x[:, None], columns=X.columns)
        line_y = model.predict(line_x_df)
        plt.plot(line_x, line_y, c="red", lw=3)
        plt.xlim(xlim); plt.ylim(ylim)
    plt.grid()
    plt.xlabel("Temperatura (°C)"); plt.ylabel("Consumi (GW)")

def main():
    
    POWER_DATA_URL = "https://github.com/datascienceunibo/dialab2024/raw/main/Regressione_non_Lineare/power.csv"

    if not os.path.exists("power.csv"):
        urlretrieve(POWER_DATA_URL, "power.csv")

    power = pd.read_csv("power.csv", index_col="date", parse_dates=["date"])
    
    
    is_train = power.index.year < 2016
    X_train = power.loc[is_train, ["temp"]]
    y_train = power.loc[is_train, "demand"]
    X_test = power.loc[~is_train, ["temp"]]
    y_test = power.loc[~is_train, "demand"]
    
    lrm = LinearRegression()
    prm = Pipeline([
        ("poly",    PolynomialFeatures(degree=30, include_bias=False)),
        ("scale",   StandardScaler()), 
        ("linreg",  LinearRegression())
    ])
    prm.fit(X_train, y_train)
    
    print_eval(X_test, y_test, prm)
    
    plot_model_on_data(X_test, y_test, prm)
    
    
    
    plt.show()

if __name__ == "__main__":
    main()
    