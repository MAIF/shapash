"""
Webapp launch module
This is an example in python how to launch app from explainer
"""
from lightgbm import LGBMClassifier, LGBMRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from shapash.explainer.smart_explainer import SmartExplainer
from category_encoders import one_hot

cases = {
    1: 'Titanic regression',
    2: 'Titanic binary classification',
    3: 'Titanic multi class classification',
}

CASE = 1

titanic = pd.read_pickle('tests/data/clean_titanic.pkl')
if CASE == 1:
    features = ['Pclass', 'Survived', 'Embarked', 'Sex']
    encoder = one_hot.OneHotEncoder(titanic, cols=['Embarked', 'Sex'])
    X = titanic[features]
    y = titanic['Age'].to_frame()
    model = LGBMRegressor()

elif CASE == 2:
    features = ['Pclass', 'Age', 'Embarked', 'Sex']
    encoder = one_hot.OneHotEncoder(titanic, cols=['Embarked', 'Sex'])
    X = titanic[features]
    y = titanic['Survived'].to_frame()
    model = LGBMClassifier()

else:
    features = ['Survived', 'Age', 'Embarked', 'Sex']
    encoder = one_hot.OneHotEncoder(titanic, cols=['Embarked', 'Sex'])
    X = titanic[features]
    y = titanic['Pclass'].to_frame()
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

xpl = SmartExplainer()
y_pred = pd.DataFrame(data=y_pred,
                      columns=y.columns.to_list(),
                      index=X_test.index)

xpl.compile(X_test, model, y_pred=y_pred, preprocessing=encoder, title_story=cases[CASE])

xpl.init_app()
app = xpl.smartapp.app

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
