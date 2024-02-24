

import statsmodels.api as sm
import statsmodels.formula.api as smf


import pandas as pd
delta_wire = pd.read_csv("delta-wire.csv")
formula = "productivity ~ training"
model_definition = smf.ols(formula, data=delta_wire)
model = model_definition.fit()


print(model.summary())



df = pd.DataFrame({"predicted": model.fittedvalues, "residuals": model.resid})
ax = df.plot.scatter(x="predicted", y="residuals")

ax.set_xlabel("Predicted productivity")
ax.set_ylabel("Residuals")
ax.axhline(0, color="black")
plt.show()


prediction = model.predict({"training": [1230, 2390]})
print(prediction)


import statsmodels.api as sm
import pandas as pd
delta_wire = pd.read_csv("delta-wire.csv")
X = delta_wire["training"]
y = delta_wire["productivity"]
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


X = delta_wire["training"]
y = delta_wire["productivity"]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


formula = "productivity ~ training - 1"
formula = "productivity ~ training + 0"


import pandas as pd
from sklearn.linear_model import LinearRegression
delta_wire = pd.read_csv("delta-wire.csv")
X = delta_wire[["training"]]
y = delta_wire["productivity"]
model = LinearRegression()
model.fit(X, y)
print(f"Model intercept: {model.intercept_}")
print(f"Model coefficients: {model.coef_}")


new_data = pd.DataFrame({'training': [1230, 2390]})
prediction = model.predict(new_data)
print(prediction)


model = LinearRegression(fit_intercept=False)
model.fit(X, y)
print(model.intercept_)
print(model.coef_)


from sklearn.model_selection import train_test_split

baseball = pd.read_csv("baseball_payroll.csv")
X = baseball[["Average Payroll (Million)"]]
y = baseball["Total Wins"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y,
test_size=0.2,
random_state=123)

model_full = LinearRegression()
model_full.fit(X, y)
model_train = LinearRegression()
model_train.fit(X_train, y_train)

print(f"Full model intercept: {model_full.intercept_}")
print(f"Full model coefficients: {model_full.coef_}")
print(f"Model intercept: {model_train.intercept_}")
print(f"Model coefficients: {model_train.coef_}")



x_range = pd.DataFrame({'Average Payroll (Million)': [0, 250]})
fig, ax = plt.subplots(figsize=[6, 4])
ax.plot(x_range, model_full.predict(x_range), color='grey')
ax.scatter(X_train, y_train, color='grey')
ax.plot(x_range, model_train.predict(x_range), color='black', linestyle='dashed')
ax.scatter(X_holdout, y_holdout, color='black', marker='x')
ax.set_xlabel("Average Payroll (Million)")
ax.set_ylabel("Total Wins")


from sklearn.metrics import mean_squared_error
rmse_full = mean_squared_error(y, model_full.predict(X), squared=False)
rmse_train = mean_squared_error(y_train, model_train.predict(X_train), squared=False)
rmse_holdout = mean_squared_error(y_holdout, model_train.predict(X_holdout), squared=False)
print(f"Full RMSE: {rmse_full:.3f}")
print(f"Training RMSE: {rmse_train:.3f}")
print(f"Holdout RMSE: {rmse_holdout:.3f}")
