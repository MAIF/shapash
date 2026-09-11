"""
Generate a house prices regression report with the smart_report implementation.

The script supports both built-in and custom report layouts:
- default_report_regression_house_prices.yml
- custom_report_regression_house_prices.yml
"""

import argparse
import os
import sys

import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading
from shapash.report.blocks import ReportBlockMixin, block


class CustomRegressionReportBlocks(ReportBlockMixin):
    """User-defined blocks for a house prices regression explainability report."""

    @block
    def block_residual_error_summary(self, title: str = "Residual error summary"):
        """Summarize residual dispersion and global regression performance indicators."""
        if self.y_test is None or self.y_pred is None:
            raise ValueError("residual_error_summary block requires y_test and y_pred.")

        y_true = pd.Series(self.y_test).reset_index(drop=True)
        y_pred = pd.Series(self.y_pred).reset_index(drop=True)
        residuals = y_true - y_pred
        abs_residuals = residuals.abs()

        summary_df = pd.DataFrame(
            [
                ["MAE", f"{mean_absolute_error(y_true, y_pred):,.2f}"],
                ["MSE", f"{mean_squared_error(y_true, y_pred):,.2f}"],
                ["R2", f"{r2_score(y_true, y_pred):.3f}"],
                ["Residual mean", f"{residuals.mean():,.2f}"],
                ["Residual std", f"{residuals.std():,.2f}"],
                ["Median absolute error", f"{abs_residuals.median():,.2f}"],
                ["95th pct absolute error", f"{abs_residuals.quantile(0.95):,.2f}"],
            ],
            columns=["Metric", "Value"],
        )

        explanation = (
            "This section summarizes global regression error levels. "
            "A strong gap between median and 95th percentile absolute error usually highlights "
            "a subset of difficult cases worth deeper explainability analysis."
        )
        return title, [explanation, summary_df]

    @block
    def block_largest_errors_focus(self, title: str = "Largest absolute errors", top_k: int = 10):
        """Display samples with the largest absolute errors to prioritize local explainability reviews."""
        explainer = self._require_explainer("largest_errors_focus")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("largest_errors_focus block requires y_test and y_pred.")

        y_true = pd.Series(self.y_test, index=explainer.x_init.index, name="true")
        y_pred = pd.Series(self.y_pred, index=explainer.x_init.index, name="pred")
        details = pd.concat([y_true, y_pred, explainer.x_init], axis=1)
        details["residual"] = details["true"] - details["pred"]
        details["abs_error"] = details["residual"].abs()

        focus = details.sort_values("abs_error", ascending=False).head(top_k).copy()
        if focus.empty:
            return title, ["No rows available to compute largest errors."]

        focus = focus.rename(columns={"true": "True", "pred": "Pred", "residual": "Residual", "abs_error": "AbsError"})

        # Keep key business columns first when available.
        preferred = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageArea", "Neighborhood"]
        context_cols = [c for c in preferred if c in focus.columns]
        leading = ["True", "Pred", "Residual", "AbsError"]
        trailing = [c for c in focus.columns if c not in leading + context_cols]
        ordered_cols = leading + context_cols + trailing

        focus = focus[ordered_cols]
        for col in ["True", "Pred", "Residual", "AbsError"]:
            focus[col] = focus[col].map(lambda x: round(float(x), 2))

        info = (
            "Rows below are the largest absolute errors. "
            "They are ideal candidates for local contribution plots and feature-level investigation."
        )
        table = focus.reset_index(drop=False)
        return title, [info, table]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate house prices regression report examples.")
    parser.add_argument(
        "--report-mode",
        choices=["default", "custom"],
        default="default",
        help="Choose report layout: default built-in blocks or custom blocks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    house_df, house_dict = data_loading("house_prices")

    y_df = house_df["SalePrice"]
    x_df = house_df[house_df.columns.difference(["SalePrice"])].copy()

    # Ensure non-numeric columns are treated as categorical before encoding.
    for col in x_df.columns:
        if not pd.api.types.is_numeric_dtype(x_df[col]):
            x_df[col] = x_df[col].astype(object)

    categorical_features = [
        col
        for col in x_df.columns
        if pd.api.types.is_object_dtype(x_df[col]) or pd.api.types.is_string_dtype(x_df[col])
    ]

    encoder = OrdinalEncoder(cols=categorical_features, handle_unknown="return_nan", return_df=True).fit(x_df)
    x_df = encoder.transform(x_df)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x_df,
        y_df,
        train_size=0.75,
        random_state=1,
    )

    regressor = RandomForestRegressor(n_estimators=200, random_state=1).fit(xtrain, ytrain)

    # Keep y_pred as dataframe to match SmartExplainer report expectations.
    y_pred = pd.DataFrame(regressor.predict(xtest), columns=["pred"], index=xtest.index)

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    xpl = SmartExplainer(
        model=regressor,
        preprocessing=encoder,
        features_dict=house_dict,
    )

    # Compile once before report generation.
    xpl.compile(x=xtest, y_pred=y_pred, y_target=ytest)

    if args.report_mode == "custom":
        output_file = os.path.join(cur_dir, "output", "regression_report_custom.html")
        report_config_file = os.path.join(cur_dir, "config", "custom_report_regression_house_prices.yml")
        block_instance = CustomRegressionReportBlocks(
            explainer=xpl,
            x_train=xtrain,
            y_train=ytrain,
            y_test=ytest,
        )
        xpl.generate_report(
            output_file=output_file,
            yaml_path=report_config_file,
            block_instance=block_instance,
        )
    else:
        output_file = os.path.join(cur_dir, "output", "regression_report.html")
        xpl.generate_report(
            output_file=output_file,
            x_train=xtrain,
            y_train=ytrain,
            y_test=ytest,
        )
