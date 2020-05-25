"""
Webapp launch module
This is an example in python how to launch app from explainer
"""
# TODO: Remove the next 4 lines, these lines allow you to run locally the code and import shapash content
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# TODO: Remove the 4 previous lines

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from shapash.data.data_loader import data_loading
from category_encoders import OrdinalEncoder
from shapash.explainer.smart_explainer import SmartExplainer

house_df, house_dict = data_loading('house_prices')
y_df=house_df['SalePrice'].to_frame()
X_df=house_df[house_df.columns.difference(['SalePrice'])]
house_df.head()

categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
encoder = OrdinalEncoder(
    cols=categorical_features,
    handle_unknown='ignore',
    return_df=True).fit(X_df)
X_df=encoder.transform(X_df)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

regressor = LGBMRegressor(n_estimators=200).fit(Xtrain,ytrain)

y_pred = pd.DataFrame(regressor.predict(Xtest),columns=['pred'],index=Xtest.index)

xpl = SmartExplainer(features_dict=house_dict) # Optional parameter, dict specifies label for features name

xpl.compile(
    x=Xtest,
    model=regressor,
    preprocessing=encoder, # Optional: compile step can use inverse_transform method
    y_pred=y_pred # Optional
)

app = xpl.run_app(port='8050')