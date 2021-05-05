"""
This script can be used to generate the report example.
For more information, please refer to the tutorial 'tuto-shapash-report01.ipynb'
that generates the same report.
"""
import os
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapash.data.data_loader import data_loading
from shapash.explainer.smart_explainer import SmartExplainer

if __name__ == "__main__":
    house_df, house_dict = data_loading('house_prices')
    y_df = house_df['SalePrice']
    X_df = house_df[house_df.columns.difference(['SalePrice'])]

    categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']

    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(X_df)

    X_df = encoder.transform(X_df)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

    regressor = RandomForestRegressor(n_estimators=50).fit(Xtrain, ytrain)

    y_pred = pd.DataFrame(regressor.predict(Xtest), columns=['pred'], index=Xtest.index)

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    xpl = SmartExplainer(features_dict=house_dict)
    xpl.compile(
        x=Xtest,
        model=regressor,
        preprocessing=encoder,  # Optional: compile step can use inverse_transform method
        y_pred=y_pred  # Optional
    )

    xpl.generate_report(
        output_file=os.path.join(cur_dir, 'output', 'report.html'),
        project_info_file=os.path.join(cur_dir, 'utils', 'project_info.yml'),
        x_train=Xtrain,
        y_train=ytrain,
        y_test=ytest,
        title_story="House prices report",
        title_description="""This document is a data science report of the kaggle house prices tutorial project. 
            It was generated using the Shapash library.""",
        metrics=[
            {
                'path': 'sklearn.metrics.mean_absolute_error',
                'name': 'Mean absolute error',
            },
            {
                'path': 'sklearn.metrics.mean_squared_error',
                'name': 'Mean squared error',
            }
        ]
    )
