"""
Webapp launch module
This is an example in python how to launch app from explainer
"""

import pandas as pd
from category_encoders import OrdinalEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading

house_df, house_dict = data_loading("house_prices")
y_df = house_df["SalePrice"].to_frame()
X_df = house_df[house_df.columns.difference(["SalePrice"])]
house_df.head()

categorical_features = [col for col in X_df.columns if X_df[col].dtype == "object"]
encoder = OrdinalEncoder(cols=categorical_features, handle_unknown="ignore", return_df=True).fit(X_df)
X_df = encoder.transform(X_df)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

regressor = LGBMRegressor(n_estimators=200).fit(Xtrain, ytrain)

y_pred = pd.DataFrame(regressor.predict(Xtest), columns=["pred"], index=Xtest.index)
y_target = pd.DataFrame(data=ytest, columns=y_df.columns.to_list(), index=Xtest.index)

xpl = SmartExplainer(
    model=regressor, features_dict=house_dict, preprocessing=encoder, title_story="House Prices - Lightgbm Regressor"
)

xpl.compile(x=Xtest, y_pred=y_pred, y_target=y_target)

xpl.init_app()
app = xpl.smartapp.app

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
