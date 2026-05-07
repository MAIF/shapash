import importlib
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import yaml
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


class NotebookParityReport(ReportBase):
    """Report with custom blocks that mirror sections from the legacy notebook report."""

    def block_project_information(self, title: str = "Project information", color: str = "gray"):
        project_info_path = self.config.get("project_info_file")
        if not project_info_path:
            raise ValueError("project_information block requires config['project_info_file'].")

        with open(project_info_path, encoding="utf-8") as f:
            project_info = yaml.safe_load(f) or {}

        sections_html = []
        for section_name, section_values in project_info.items():
            if not isinstance(section_values, dict):
                continue
            if section_name.strip().lower() == "model training":
                continue

            rows = self._render_key_value_rows(section_values)
            sections_html.append(
                f'<div class="content-block"><h3 class="section-title" style="font-size:1.2rem">{section_name}</h3>'
                f'<table class="kv-table">{rows}</table></div>'
            )

        return self._wrap_section_content(title, "".join(sections_html))

    def block_model_analysis(self, title: str = "Model information", color: str = "blue"):
        explainer = self._require_explainer("model_analysis")
        model = explainer.model
        model_name = type(model).__name__
        details = {
            "Model class": model_name,
            "Task": getattr(explainer, "_case", "regression"),
            "Feature count": len(explainer.x_init.columns),
            "Prediction sample size": len(explainer.x_init),
            "Training sample size": len(self.x_train_init) if self.x_train_init is not None else "n/a",
        }
        rows = self._render_key_value_rows(details)
        return self._wrap_section_content(title, f'<table class="kv-table">{rows}</table>')

    def block_relationship_target(
        self,
        title: str = "Relationship with target variable",
        feature: str = "OverallQual",
        color: str = "blue",
        max_y: int | None = None,
    ):
        self._require_train_test_data("relationship_target")
        if self.x_train_pre is None or self.y_train is None:
            raise ValueError("relationship_target block requires both training features and y_train.")
        if feature not in self.x_train_pre.columns:
            raise ValueError(f"Unknown feature '{feature}' for relationship_target block.")

        target_name = self.target_name_train or "target"
        df_train = self.x_train_pre.copy()
        df_train[target_name] = self.y_train

        fig = px.box(df_train, x=feature, y=target_name)
        if max_y is not None:
            fig.update_yaxes(range=[0, max_y])
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_training_correlations(
        self,
        title: str = "Relationship between training variables",
        color: str = "blue",
        max_features: int = 30,
    ):
        if self.x_train_pre is None:
            raise ValueError("training_correlations block requires x_train.")

        numeric_train = self.x_train_pre.select_dtypes(include="number")
        corr = numeric_train.corr(numeric_only=True)
        if max_features > 0 and corr.shape[0] > max_features:
            corr = corr.iloc[:max_features, :max_features]

        fig = px.imshow(corr, color_continuous_scale="YlGnBu", zmin=-1, zmax=1, aspect="auto")
        return self._wrap_section_content(title, self._plotly_html(fig))

    def block_performance_metrics(
        self,
        title: str = "Model performance",
        color: str = "orange",
        metrics: list | None = None,
    ):
        if self.y_test is None or self.y_pred is None:
            raise ValueError("performance_metrics block requires y_test and y_pred.")

        metric_items = []
        metrics = metrics or []
        for metric_cfg in metrics:
            metric_path = metric_cfg.get("path")
            metric_name = metric_cfg.get("name", metric_path)
            if not metric_path:
                continue
            module_path, fn_name = metric_path.rsplit(".", 1)
            metric_fn = getattr(importlib.import_module(module_path), fn_name)
            value = metric_fn(self.y_test, self.y_pred)
            metric_items.append({"label": metric_name, "value": f"{value:,.2f}", "color": color})

        return self.block_badge_row(title=title, badges=metric_items)

    def block_pred_vs_true(self, title: str = "y_pred vs y_test", color: str = "orange"):
        if self.y_test is None or self.y_pred is None:
            raise ValueError("pred_vs_true block requires y_test and y_pred.")

        scatter_df = pd.DataFrame({"y_test": self.y_test, "y_pred": self.y_pred})
        fig = px.scatter(scatter_df, x="y_test", y="y_pred")
        return self._wrap_section_content(title, self._plotly_html(fig))


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

    report = NotebookParityReport(
        explainer=xpl,
        x_train=Xtrain,
        y_train=ytrain,
        y_test=ytest,
        config={"project_info_file": str(PROJECT_INFO_FILE)},
    )
    report.generate_report(config_file=str(CONFIG_V1), output_file=str(OUTPUT_V1))
    print(f"Saved notebook-parity report: {OUTPUT_V1}")