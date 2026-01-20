"""
Webapp launch module
This is an example in python how to launch app from explainer
"""

import pandas as pd
from category_encoders import one_hot
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading

LANG = "FR"  # "FR" or "EN"
CASE = 2  # 1: Regression, 2: Binary classification, 3: Multi class classification

if LANG == "FR":
    cases = {
        1: "Titanic Régression (Prix du billet)",
        2: "Titanic Classification binaire (Survie)",
        3: "Titanic Multi-classes (Classe du billet)",
    }
else:
    cases = {
        1: "Titanic Regression (Ticket Fare)",
        2: "Titanic Binary classification (Survival)",
        3: "Titanic Multi class (Ticket Class)",
    }

titanic_df, titanic_dict = data_loading("titanic")

titanic_df["Pclass"] = titanic_df["Pclass"].map({"First class": 1, "Second class": 2, "Third class": 3})

if LANG == "FR":
    feature_dict = {
        "Age": "Âge",
        "Embarked": "Porte d'embarquement",
        "Fare": "Prix du billet",
        "Name": "Nom",
        "Parch": "Nombre de parents/enfants à bord",
        "Pclass": "Classe du billet",
        "Sex": "Sexe",
        "SibSp": "Nombre de frères et sœurs/conjoints à bord",
        "Survived": "Survie",
        "Title": "Titre",
    }
    postprocess = {
        "Age": {
            "type": "suffix",
            "rule": " ans",  # Adding 'ans' at the end
        },
        "Sex": {"type": "transcoding", "rule": {"male": "Homme", "female": "Femme"}},
        "Fare": {
            "type": "prefix",
            "rule": "$",  # Adding $ at the beginning
        },
        "Embarked": {"type": "case", "rule": "upper"},
        "Pclass": {
            "type": "transcoding",
            "rule": {1: "1ère classe", 2: "2ème classe", 3: "3ème classe"},
        },
        "Survived": {
            "type": "transcoding",
            "rule": {0: "Décédé", 1: "Survivant"},
        },
    }
else:
    feature_dict = {
        "Parch": "Number of parents/children aboard",
        "Pclass": "Ticket class",
        "SibSp": "Number of siblings/spouses aboard",
    }
    postprocess = {
        "Age": {
            "type": "suffix",
            "rule": " years old",  # Adding 'years old' at the end
        },
        "Sex": {"type": "transcoding", "rule": {"male": "Man", "female": "Woman"}},
        "Fare": {
            "type": "prefix",
            "rule": "$",  # Adding $ at the beginning
        },
        "Embarked": {"type": "case", "rule": "upper"},
        "Pclass": {
            "type": "transcoding",
            "rule": {1: "First", 2: "Second", 3: "Third"},
        },
        "Survived": {
            "type": "transcoding",
            "rule": {0: "Deceased", 1: "Survived"},
        },
    }

if CASE == 1:
    features = ["Pclass", "Survived", "Embarked", "Sex", "Age", "SibSp", "Parch"]
    for col in list(feature_dict.keys()):
        if col not in features:
            del feature_dict[col]
    for col in list(postprocess.keys()):
        if col not in features:
            del postprocess[col]
    encoder = one_hot.OneHotEncoder(titanic_df, cols=["Embarked", "Sex"])

    X = titanic_df[features]
    y = titanic_df["Fare"].to_frame()

    model = LGBMRegressor(verbose=-1)
    response_dict = None
    additional_data = titanic_df[["Name", "Title"]]
    if LANG == "FR":
        additional_features_dict = {"Name": "Nom", "Title": "Titre"}
    else:
        additional_features_dict = None

elif CASE == 2:
    features = ["Pclass", "Age", "Sex", "SibSp", "Parch"]
    for col in list(feature_dict.keys()):
        if col not in features:
            del feature_dict[col]
    for col in list(postprocess.keys()):
        if col not in features:
            del postprocess[col]
    encoder = one_hot.OneHotEncoder(titanic_df, cols=["Sex"])

    X = titanic_df[features]
    y = titanic_df["Survived"].to_frame()
    model = LGBMClassifier(max_depth=3, verbose=-1, n_estimators=100)

    additional_data = titanic_df[["Name", "Fare", "Title", "Embarked"]]
    if LANG == "FR":
        response_dict = {0: "Décédé", 1: "Survivant"}
        additional_features_dict = {
            "Name": "Nom",
            "Title": "Titre",
            "Fare": "Prix du billet",
            "Embarked": "Porte d'embarquement",
        }
    else:
        response_dict = {0: "Deceased", 1: "Survived"}
        additional_features_dict = None

else:
    features = ["Survived", "Age", "Embarked", "Sex"]
    for col in list(feature_dict.keys()):
        if col not in features:
            del feature_dict[col]
    for col in list(postprocess.keys()):
        if col not in features:
            del postprocess[col]

    encoder = one_hot.OneHotEncoder(titanic_df, cols=["Embarked", "Sex"])

    X = titanic_df[features]
    y = titanic_df["Pclass"].to_frame()
    model = LGBMClassifier(verbose=-1)

    additional_data = titanic_df[["Name", "Fare", "Title"]]
    if LANG == "FR":
        response_dict = {1: "1ère classe", 2: "2ème classe", 3: "3ème classe"}
        additional_features_dict = {"Name": "Nom", "Title": "Titre", "Fare": "Prix du billet"}
    else:
        response_dict = {1: "First Class", 2: "Second Class", 3: "Third Class"}
        additional_features_dict = None

titanic_enc = encoder.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(titanic_enc, y, test_size=0.2, random_state=79)


df = titanic_df[features + y.columns.to_list()]
df = df.loc[titanic_enc.index, :]
df.reset_index(level=0, inplace=True)

model.fit(X_train, y_train)

# -----------------------------
# Prédictions
# -----------------------------
if CASE == 1:
    y_train_ = y_train.iloc[:, 0]
    y_test_ = y_test.iloc[:, 0]
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    df_metrics = pd.DataFrame(
        {
            "rmse": [
                ((y_train_ - y_pred_train) ** 2).mean() ** 0.5,
                ((y_test_ - y_pred_test) ** 2).mean() ** 0.5,
            ],
            "mae": [
                (y_train_ - y_pred_train).abs().mean(),
                (y_test_ - y_pred_test).abs().mean(),
            ],
        },
        index=["train", "test"],
    )
else:
    y_train_ = y_train.iloc[:, 0]
    y_test_ = y_test.iloc[:, 0]
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    is_binary = y_train_.nunique() == 2
    avg = "binary" if is_binary else "macro"

    df_metrics = pd.DataFrame(
        {
            "accuracy": [
                accuracy_score(y_train_, y_pred_train),
                accuracy_score(y_test_, y_pred_test),
            ],
            "precision": [
                precision_score(y_train_, y_pred_train, average=avg),
                precision_score(y_test_, y_pred_test, average=avg),
            ],
            "recall": [
                recall_score(y_train_, y_pred_train, average=avg),
                recall_score(y_test_, y_pred_test, average=avg),
            ],
            "f1_score": [
                f1_score(y_train_, y_pred_train, average=avg),
                f1_score(y_test_, y_pred_test, average=avg),
            ],
        },
        index=["train", "test"],
    )


y_pred = model.predict(titanic_enc)

y_pred = pd.DataFrame(data=y_pred, columns=y.columns.to_list(), index=titanic_enc.index)

y_target = pd.DataFrame(data=y, columns=y.columns.to_list(), index=titanic_enc.index)


# -----------------------------
# Tableau récapitulatif
# -----------------------------
if CASE == 1:
    print(f"Mean Fare: {y_target.iloc[:,0].mean():.2f}")
elif CASE == 2:
    print(f"Survival Rate: {y_target.iloc[:,0].mean():.2f}")
else:
    print(f"Class distribution: {y_target.iloc[:,0].value_counts(normalize=True)}")

print(df_metrics)

xpl = SmartExplainer(
    model,
    preprocessing=encoder,
    postprocessing=postprocess,
    title_story=cases[CASE],
    label_dict=response_dict,
    features_dict=feature_dict,
)

xpl.compile(
    titanic_enc,
    y_pred=y_pred,
    y_target=y_target,
    additional_data=additional_data.loc[titanic_enc.index, :],
    additional_features_dict=additional_features_dict,
)

xpl.init_app()
app = xpl.smartapp.app

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
