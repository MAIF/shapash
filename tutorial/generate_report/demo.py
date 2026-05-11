import sys
from pathlib import Path

import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading
from shapash.report.smart_report import ReportBase

CONFIG_V1 = HERE / "report_config_v1.yml"
OUTPUT_V1 = HERE / "output" / "report_v1.html"
PROJECT_INFO_FILE = HERE / "utils" / "project_info.yml"


def build_house_prices_explainer() -> tuple[SmartExplainer, pd.DataFrame, pd.Series, pd.Series]:
    """Build the same House Prices explainer used in report tutorials."""
    house_df, house_dict = data_loading("house_prices")
    y_df = house_df["SalePrice"]
    X_df = house_df[house_df.columns.difference(["SalePrice"])]

    categorical_features = list(X_df.select_dtypes(include=["object", "string", "category"]).columns)
    encoder = OrdinalEncoder(cols=categorical_features, handle_unknown="ignore", return_df=True).fit(X_df)
    X_encoded = encoder.transform(X_df)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_encoded, y_df, train_size=0.75, random_state=1)
    regressor = RandomForestRegressor(n_estimators=50, random_state=1).fit(Xtrain, ytrain)

    y_pred = pd.DataFrame(regressor.predict(Xtest), columns=["pred"], index=Xtest.index)

    xpl = SmartExplainer(model=regressor, preprocessing=encoder, features_dict=house_dict)
    xpl.compile(x=Xtest, y_pred=y_pred, y_target=ytest)
    return xpl, Xtrain, ytrain, ytest


if __name__ == "__main__":
    xpl, Xtrain, ytrain, ytest = build_house_prices_explainer()

    report = ReportBase(
        explainer=xpl,
        x_train=Xtrain,
        y_train=ytrain,
        y_test=ytest,
        config={"project_info_file": str(PROJECT_INFO_FILE)},
    )
    report.generate_report(config_file=str(CONFIG_V1), output_file=str(OUTPUT_V1))
    print(f"Saved notebook-parity report: {OUTPUT_V1}")
