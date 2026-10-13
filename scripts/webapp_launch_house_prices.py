"""
Webapp launch script
Example of how to launch the webapp from an explainer, on the house prices dataset.

Usage:
    python scripts/webapp_launch_house_prices.py --port 8080
"""

import argparse

import pandas as pd
from category_encoders import OrdinalEncoder
from dash import Dash
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading
from shapash.explainer.smart_explainer import DEFAULT_HOST
from shapash.utils.dtypes import text_like_columns


def build_app() -> Dash:
    """Train a house prices regressor, compile a SmartExplainer and return its Dash app."""
    house_df, house_dict = data_loading("house_prices")
    y_df = house_df["SalePrice"].to_frame()
    X_df = house_df[house_df.columns.difference(["SalePrice"])]

    categorical_features = text_like_columns(X_df, strict_object=False)
    encoder = OrdinalEncoder(cols=categorical_features, handle_unknown="return_nan", return_df=True).fit(X_df)
    X_df = encoder.transform(X_df)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

    regressor = LGBMRegressor(n_estimators=200).fit(Xtrain, ytrain)

    y_pred = pd.DataFrame(regressor.predict(Xtest), columns=["pred"], index=Xtest.index)
    y_target = pd.DataFrame(data=ytest, columns=y_df.columns.to_list(), index=Xtest.index)

    xpl = SmartExplainer(
        model=regressor,
        features_dict=house_dict,
        preprocessing=encoder,
        title_story="House Prices - Lightgbm Regressor",
    )

    xpl.compile(x=Xtest, y_pred=y_pred, y_target=y_target)

    xpl.init_app()
    return xpl.smartapp.app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch the Shapash webapp on the house prices dataset.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true", help="run Dash in debug mode")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build the app and serve it."""
    args = parse_args(argv)
    app = build_app()
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
