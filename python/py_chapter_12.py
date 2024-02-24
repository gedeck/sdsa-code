

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

mower_df = pd.read_csv("RidingMowers.csv")
outcome = "Ownership"
predictors = ["Income", "Lot_Size"]

X = mower_df[predictors]
y = mower_df[outcome]

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_normalized, y)


new_customer = pd.DataFrame({"Income": 60, "Lot_Size": 20},
                        index=["New customer"])
new_customer_normalized = scaler.transform(new_customer)
pred_class = model.predict(new_customer_normalized)
print(f'Class predicted for the the new customer: {pred_class[0]}')


pred_class = model.predict_proba(new_customer_normalized)
print(f'Class predicted for the the new customer: {pred_class[0]}')


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(mower_df[outcome], model.predict(X_normalized))
print(f'Accuracy: {accuracy:.2f}')


from sklearn.pipeline import Pipeline
model = Pipeline(steps=[
    ('normalize', StandardScaler()),
    ('kNN', KNeighborsClassifier(n_neighbors=5))
])
print(model)


model.fit(X, y)


pred_class = model.predict(new_customer)
print(f'Class predicted for the the new customer: {pred_class[0]}')


accuracy = accuracy_score(mower_df[outcome], model.predict(X))
print(f'Accuracy: {accuracy:.2f}')
