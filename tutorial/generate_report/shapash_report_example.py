"""
Generate the report example with the new smart_report implementation.

The report layout is driven by the YAML file `default_report.yml` and rendered
through `SmartExplainer.generate_report`.

For more information, please refer to the tutorial
`tuto-shapash-report01.ipynb` that generates the same report.
"""
import os
import sys

import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading
from shapash.report.smart_report.blocks import ReportBlockMixin

# Custom block class can be defined by inheriting from ReportBlockMixin and implementing block methods.
class UserReportBlocks(ReportBlockMixin):
    """Example of user-defined custom blocks for report generation."""

    def block_user_note(
        self,
        title: str = "Analyst note",
        body: str = "This report includes a custom user cell.",
    ) -> str:
        return self.block_text(title=title, body=body)

if __name__ == "__main__":
    house_df, house_dict = data_loading("house_prices")
    y_df = house_df["SalePrice"]
    X_df = house_df[house_df.columns.difference(["SalePrice"])].copy()

    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].astype(object)

    categorical_features = [col for col in X_df.columns if X_df[col].dtype == "object"]

    encoder = OrdinalEncoder(cols=categorical_features, handle_unknown="ignore", return_df=True).fit(X_df)

    X_df = encoder.transform(X_df)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

    regressor = RandomForestRegressor(n_estimators=50).fit(Xtrain, ytrain)

    y_pred = pd.DataFrame(regressor.predict(Xtest), columns=["pred"], index=Xtest.index)

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    xpl = SmartExplainer(
        model=regressor,
        preprocessing=encoder,  # Optional: compile step can use inverse_transform method
        features_dict=house_dict,
    )
    xpl.compile(x=Xtest, y_pred=y_pred, y_target=ytest)

    output_file = os.path.join(cur_dir, "output", "report.html")
    project_info_file = os.path.join(cur_dir, "config", "project_information.yml")
    custom_report_config_file = os.path.join(cur_dir, "config", "default_report_custom.yml")

    xpl.generate_report(
        output_file=output_file,
        project_info_file=project_info_file,
        x_train=Xtrain,
        y_train=ytrain,
        y_test=ytest,
        yaml_path=custom_report_config_file,
        # Use the custom block class to enable user-defined blocks in the report.
        block_class=UserReportBlocks,
    )
