import importlib
import sys
from pathlib import Path

import pandas as pd
import yaml
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, HoverTool, LinearColorMapper, Span
from bokeh.palettes import RdYlBu11
from bokeh.plotting import figure
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

        grouped = df_train.groupby(feature)[target_name]
        q1 = grouped.quantile(0.25)
        q2 = grouped.quantile(0.5)
        q3 = grouped.quantile(0.75)
        iqr = q3 - q1
        upper = (q3 + 1.5 * iqr).clip(upper=grouped.max())
        lower = (q1 - 1.5 * iqr).clip(lower=grouped.min())

        cats = [str(c) for c in q1.index.tolist()]
        source = ColumnDataSource(
            data={
                "cat": cats,
                "q1": q1.values,
                "q2": q2.values,
                "q3": q3.values,
                "upper": upper.values,
                "lower": lower.values,
            }
        )

        p = figure(
            title=title,
            x_range=cats,
            width=900,
            height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.segment("cat", "upper", "cat", "q3", source=source, line_color="#444444")
        p.segment("cat", "lower", "cat", "q1", source=source, line_color="#444444")
        p.vbar("cat", 0.7, "q2", "q3", source=source, fill_color="#9ecae1", line_color="#2b8cbe")
        p.vbar("cat", 0.7, "q1", "q2", source=source, fill_color="#fdd0a2", line_color="#d95f0e")
        p.rect("cat", "lower", 0.2, 0.001, source=source, line_color="#444444")
        p.rect("cat", "upper", 0.2, 0.001, source=source, line_color="#444444")
        p.add_tools(
            HoverTool(
                tooltips=[
                    (feature, "@cat"),
                    ("Q1", "@q1{0,0.00}"),
                    ("Median", "@q2{0,0.00}"),
                    ("Q3", "@q3{0,0.00}"),
                    ("Lower", "@lower{0,0.00}"),
                    ("Upper", "@upper{0,0.00}"),
                ]
            )
        )
        p.xaxis.major_label_orientation = 0.8
        p.xaxis.axis_label = feature
        p.yaxis.axis_label = target_name
        if max_y is not None:
            p.y_range.end = max_y
        return self._wrap_section_content("", self._bokeh_html(p))

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

        corr = corr.fillna(0.0)
        x_labels = list(corr.columns)
        y_labels = list(corr.index)
        corr_long = (
            corr.stack()
            .rename("corr")
            .reset_index()
            .rename(columns={"level_0": "y", "level_1": "x"})
        )
        source = ColumnDataSource(corr_long)

        color_mapper = LinearColorMapper(palette=list(reversed(RdYlBu11)), low=-1, high=1)
        p = figure(
            title=title,
            x_range=x_labels,
            y_range=list(reversed(y_labels)),
            width=950,
            height=650,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="right",
        )
        renderer = p.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=source,
            line_color=None,
            fill_color={"field": "corr", "transform": color_mapper},
        )

        p.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[("Feature X", "@x"), ("Feature Y", "@y"), ("Correlation", "@corr{0.000}")],
            )
        )
        p.xaxis.major_label_orientation = 0.9
        p.grid.grid_line_color = None

        color_bar = ColorBar(
            color_mapper=color_mapper,
            ticker=BasicTicker(desired_num_ticks=7),
            label_standoff=8,
            location=(0, 0),
        )
        p.add_layout(color_bar, "right")
        return self._wrap_section_content("", self._bokeh_html(p))

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

    def block_feature_importance(self, title: str = "Model explainability", color: str = "gold", label=None):
        explainer = self._require_explainer("feature_importance")
        if getattr(explainer, "features_imp", None) is None:
            explainer.compute_features_import()

        features_imp = explainer.features_imp
        if isinstance(features_imp, list):
            if not features_imp:
                raise ValueError("features_imp is empty.")
            features_imp = features_imp[0]

        top_n = 20
        ordered = features_imp.sort_values(ascending=False).head(top_n)
        display_names = [explainer.features_dict.get(name, name) for name in ordered.index.tolist()]
        source = ColumnDataSource(
            data={
                "feature": list(reversed(display_names)),
                "importance": list(reversed(ordered.values.tolist())),
            }
        )

        p = figure(
            title=title,
            y_range=list(reversed(display_names)),
            width=900,
            height=560,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        renderer = p.hbar(y="feature", right="importance", height=0.7, source=source, color="#ffbb00")
        p.xaxis.axis_label = "Importance"
        p.yaxis.axis_label = "Feature"
        p.grid.grid_line_alpha = 0.25
        p.add_tools(HoverTool(renderers=[renderer], tooltips=[("Feature", "@feature"), ("Importance", "@importance{0.00}")]))
        return self._wrap_section_content("", self._bokeh_html(p))

    def block_pred_vs_true(self, title: str = "y_pred vs y_test", color: str = "orange"):
        if self.y_test is None or self.y_pred is None:
            raise ValueError("pred_vs_true block requires y_test and y_pred.")

        scatter_df = pd.DataFrame({"y_test": self.y_test, "y_pred": self.y_pred})
        source = ColumnDataSource(scatter_df)
        min_v = float(min(scatter_df["y_test"].min(), scatter_df["y_pred"].min()))
        max_v = float(max(scatter_df["y_test"].max(), scatter_df["y_pred"].max()))
        p = figure(title=title, width=900, height=500, tools="pan,wheel_zoom,box_zoom,reset,save")
        p.scatter("y_test", "y_pred", source=source, size=7, alpha=0.6, color="#2255aa")
        ref_line = Span(location=0, dimension="width")
        p.renderers.append(ref_line)
        p.line([min_v, max_v], [min_v, max_v], line_dash="dashed", color="#777777", line_width=2)
        p.xaxis.axis_label = "y_test"
        p.yaxis.axis_label = "y_pred"
        p.add_tools(HoverTool(tooltips=[("y_test", "@y_test{0,0.00}"), ("y_pred", "@y_pred{0,0.00}")]))
        return self._wrap_section_content("", self._bokeh_html(p))


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
