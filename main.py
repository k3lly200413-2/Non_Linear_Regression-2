import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from urllib.request import urlretrieve
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

POWER_DATA_URL = "https://github.com/datascienceunibo/dialab2024/raw/main/Regressione_non_Lineare/power.csv"
HOUSING_DATA_URL = "https://github.com/datascienceunibo/dialab2024/raw/main/Regressione_non_Lineare/housing.csv"

if not os.path.exists("power.csv"):
    urlretrieve(POWER_DATA_URL, "power.csv")

power = pd.read_csv("power.csv", index_col="date", parse_dates=["date"])
    
is_train = power.index.year < 2016
X_train = power.loc[is_train, ["temp"]]
y_train = power.loc[is_train, "demand"]
X_test = power.loc[~is_train, ["temp"]]
y_test = power.loc[~is_train, "demand"]

def test_regression(degree, alpha):
    global X_train, y_train
    prm = Pipeline([
        ("poly",    PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale",   StandardScaler()),
        ("linreg",  Ridge(alpha=alpha))
    ])
    
    prm.fit(X_train, y_train)
    return prm.score(X_test, y_test)

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
    global X_train, y_train
    
    if not os.path.exists("housing.csv"):
        urlretrieve(HOUSING_DATA_URL, "housing.csv")

    housing = pd.read_csv("housing.csv")
    
    y = housing["MEDV"]
    X = housing.drop(columns="MEDV")
    housing = pd.read_csv("housing.csv")
    
    lrm = LinearRegression()
    prm = Pipeline([
        ("poly",    PolynomialFeatures(degree=30, include_bias=False)),
        ("scale",   StandardScaler()), 
        ("linreg",  Ridge(alpha=1))
    ])
    prm.fit(X_train, y_train)
    
    # print_eval(X_test, y_test, prm)
    
    # plot_model_on_data(X_test, y_test, prm)
    
    res_degree  = np.arange(3, 30)
    
    res_low_reg  = np.array([test_regression(d, 0.01) for d in res_degree ])
    
    # print(result)
    
    res_high_reg  = np.array([test_regression(d, 10) for d in res_degree ])
    
    # print(res_high_reg )
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(res_degree, res_low_reg, "ro-")
    # plt.plot(res_degree, res_high_reg, "bo-")
    # plt.grid()
    # plt.xlabel("Grado regr. polinomiale")
    # plt.ylabel("Score R²")
    # # aggiungiamo una legenda al grafico
    # plt.legend(["α = 0.01", "α = 10"], loc="lower right")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    
    # model_a = LinearRegression()
    # model_b = Ridge(alpha=10)
    # model_c = Pipeline([
    #     ("stand",   StandardScaler()),
    #     ("linreg",  LinearRegression())
    # ])
    # model_c.fit(X_train, y_train)
    # print_eval(X_train, y_train, model_c)
    # model_b.fit(X_train, y_train)
    # print_eval(X_train, y_train, model_b)
    # model_a.fit(X_train, y_train)
    # print_eval(X_train, y_train, model_a)
    
    # print(pd.DataFrame({
    #     "linear": model_a.coef_,
    #     "ridge": model_b.coef_,
    #     "scaled": model_c.named_steps["linreg"].coef_
    # }, index=X.columns))
    
    # model = Pipeline([
    #     ("scale", StandardScaler()),
    #     ("regr", Lasso(alpha=0.2))
    # ])
    # model.fit(X_train, y_train)
    
    # print_eval(X_test, y_test, model)
    
    model = Pipeline([
        ("scale",   StandardScaler()),
        ("reg",     ElasticNet(alpha=0.2, l1_ratio=0.1))
    ])
    
    model.fit(X_train, y_train)
    print_eval(X_train, y_train, model)
    
    plt.show()

if __name__ == "__main__":
    main()
    