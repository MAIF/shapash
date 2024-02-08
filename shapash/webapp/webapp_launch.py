"""
Webapp launch module
This is an example in python how to launch app from explainer
"""
import pandas as pd
from category_encoders import one_hot
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

from shapash import SmartExplainer

cases = {
    1: "Titanic regression",
    2: "Titanic binary classification",
    3: "Titanic multi class classification",
}

CASE = 2

titanic = pd.read_pickle("tests/data/clean_titanic.pkl")

if CASE == 1:
    features = ["Pclass", "Survived", "Embarked", "Sex"]
    encoder = one_hot.OneHotEncoder(titanic, cols=["Embarked", "Sex"])
    X = titanic[features]
    y = titanic["Age"].to_frame()
    model = LGBMRegressor()

elif CASE == 2:
    features = ["Pclass", "Age", "Embarked", "Sex"]
    encoder = one_hot.OneHotEncoder(titanic, cols=["Embarked", "Sex"])
    X = titanic[features]
    y = titanic["Survived"].to_frame()
    model = LGBMClassifier()

else:
    features = ["Survived", "Age", "Embarked", "Sex"]
    encoder = one_hot.OneHotEncoder(titanic, cols=["Embarked", "Sex"])
    X = titanic[features]
    y = titanic["Pclass"].to_frame()
    model = LGBMClassifier()

titanic_enc = encoder.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    titanic_enc,
    y,
    test_size=0.2,
)

X_test_ini = X.loc[X_test.index, :]

df = titanic[features + y.columns.to_list()]
df = df.loc[X_test.index, :]
df.reset_index(level=0, inplace=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = pd.DataFrame(data=y_pred, columns=y.columns.to_list(), index=X_test.index)

y_target = pd.DataFrame(data=y_test, columns=y.columns.to_list(), index=X_test.index)

response_dict = {0: "Death", 1: "Survival"}
xpl = SmartExplainer(model, preprocessing=encoder, title_story=cases[CASE], label_dict=response_dict)

xpl.compile(X_test, y_pred=y_pred, y_target=y_target)

xpl.init_app()
app = xpl.smartapp.app

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
