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
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading
from shapash.report.blocks import ReportBlockMixin, block

# Custom block class can be defined by inheriting from ReportBlockMixin and implementing block methods.
class UserReportBlocks(ReportBlockMixin):
    """Example of user-defined custom blocks for report generation."""

    @block
    def block_user_note(
        self,
        title: str = "Analyst note",
        body: str = "This report includes a custom user cell.",
    ) -> str:
        return title, [pn.pane.Markdown(body)]

    @block
    def block_prediction_diagnostics(
        self,
        title: str = "Prediction diagnostics",
        color_feature: str | None = None,
    ) -> str:
        """Display a richer custom block with complementary prediction graphs."""
        if self.y_test is None or self.y_pred is None:
            return title, [pn.pane.Markdown("Prediction diagnostics requires both y_test and y_pred.")]

        diagnostics = pd.DataFrame(
            {
                "actual": pd.Series(self.y_test).reset_index(drop=True),
                "predicted": pd.Series(self.y_pred).reset_index(drop=True),
            }
        )
        diagnostics["residual"] = diagnostics["actual"] - diagnostics["predicted"]
        diagnostics["abs_error"] = diagnostics["residual"].abs()

        if color_feature and self.x_init is not None and color_feature in self.x_init.columns:
            diagnostics[color_feature] = pd.Series(self.x_init[color_feature]).reset_index(drop=True)
            scatter = px.scatter(
                diagnostics,
                x="actual",
                y="predicted",
                color=color_feature,
                hover_data=["residual", "abs_error"],
                title="Actual vs predicted",
                labels={"actual": "Actual", "predicted": "Predicted"},
            )
        else:
            scatter = px.scatter(
                diagnostics,
                x="actual",
                y="predicted",
                color="abs_error",
                color_continuous_scale="Tealgrn",
                hover_data=["residual", "abs_error"],
                title="Actual vs predicted",
                labels={"actual": "Actual", "predicted": "Predicted", "abs_error": "Absolute error"},
            )

        min_axis = min(diagnostics["actual"].min(), diagnostics["predicted"].min())
        max_axis = max(diagnostics["actual"].max(), diagnostics["predicted"].max())
        scatter.add_trace(
            go.Scatter(
                x=[min_axis, max_axis],
                y=[min_axis, max_axis],
                mode="lines",
                line={"dash": "dash", "color": "#666666"},
                name="Ideal fit",
                showlegend=False,
            )
        )
        scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))

        residual_hist = px.histogram(
            diagnostics,
            x="residual",
            nbins=30,
            title="Residual distribution",
            labels={"residual": "Actual - Predicted"},
            color_discrete_sequence=["#2E8B57"],
        )
        residual_hist.add_vline(x=0, line_dash="dash", line_color="#666666")
        residual_hist.update_layout(margin=dict(l=20, r=20, t=50, b=20))

        summary = pn.pane.Markdown(
            (
                "**Quick diagnostics:** "
                f"MAE = {diagnostics['abs_error'].mean():.2f}, "
                f"mean residual = {diagnostics['residual'].mean():.2f}, "
                f"max absolute error = {diagnostics['abs_error'].max():.2f}"
            )
        )

        # Avoid stretch_both here: Panel standalone export can compute a collapsed row
        # height, which makes the next section overlap these figures.
        scatter_pane = pn.pane.Plotly(
            scatter,
            config={"displayModeBar": False, "responsive": True},
            sizing_mode="stretch_width",
            height=360,
        )
        residual_pane = pn.pane.Plotly(
            residual_hist,
            config={"displayModeBar": False, "responsive": True},
            sizing_mode="stretch_width",
            height=360,
        )
        charts = pn.Row(scatter_pane, residual_pane, sizing_mode="stretch_width")
        return title, [summary, charts]

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
        block_instance=UserReportBlocks(),
    )
